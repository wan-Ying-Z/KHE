import gzip
import numpy as np
import torch
import rdkit
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils import data

from gnn.graph_features import atom_features
from collections import defaultdict
from rdkit.Chem import AllChem
from torch.utils import data



class FeatureDataset(data.Dataset):
    def __init__(self,drugpairs_path):

        print(drugpairs_path)
        drug_combo_file = drugpairs_path
        drug_momo_file = './DB_data/node_fts/drug_momo_fts.csv'
        drug_protein_file = './DB_data/node_fts/drug_protein_fts.csv'
        drug_smile_file = './DB_data/node_fts/drug_smiles.csv'

        ####combo
        self.drug1 = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[0], dtype=np.float, comments=None)
        self.drug2 = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[1], dtype=np.float, comments=None)
        ###SE
        self.side_e=np.genfromtxt(drug_combo_file, delimiter=',', usecols=[2], dtype=np.float, comments=None)
        ###target
        self.target = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[3], dtype=np.float,
                                    comments=None)

        ####单药副作用
        self.drug_info = np.genfromtxt(drug_momo_file, delimiter=',', usecols=[0], dtype=np.float, comments=None)
        self.drug_momo_fts = np.genfromtxt(drug_momo_file, delimiter=',', usecols=range(1, 9263), dtype=np.float,
                                           comments=None)

        ###蛋白质特征
        self.drug_protein_fts = np.genfromtxt(drug_protein_file, delimiter=',', usecols=range(1, 1034), dtype=np.float,
                                              comments=None)

        ###药物smiles
        self.drug_smile_fts = np.genfromtxt(drug_smile_file, skip_header=1, delimiter=',', usecols=range(3, 4),
                                            dtype=np.str_, comments=None)



    def __getitem__(self, index):
        ######这里要返回两个药的所有特征


        d1_info = int(self.drug1[index])
        d1_momo_fts = self.drug_momo_fts[d1_info]
        d1_protein_fts = self.drug_protein_fts[d1_info]
        d1_fingerprint = smile_to_fingerprint(self.drug_smile_fts[d1_info])

        d2_info = int(self.drug2[index])
        d2_momo_fts = self.drug_momo_fts[d2_info]
        d2_protein_fts = self.drug_protein_fts[d2_info]
        d2_fingerprint = smile_to_fingerprint(self.drug_smile_fts[d2_info])

        target = self.target[index]
        side_e=self.side_e[index]

        #####单药副作用，蛋白质特征，指纹特征
        return d1_fingerprint,d2_fingerprint,target,side_e
        #return d1_momo_fts, d1_protein_fts, d1_fingerprint, d2_momo_fts, d2_protein_fts, d2_fingerprint, target

    def __len__(self):
        return len(self.drug1)


def Fts_collate_fn(data):
    n_samples = len(data)
    d1_momo_fts,d1_protein_fts,d1_fingerprint,d2_momo_fts,d2_protein_fts,d2_fingerprint,target=data[0]

    n_momo=len(d1_momo_fts)
    n_protein=len(d1_protein_fts)
    n_smile=len(d1_fingerprint)
    n_target = len(target)

    d1_momo_tensor = torch.zeros(n_samples, n_momo)
    d1_protein_tensor = torch.zeros(n_samples, n_protein)
    d1_smile_tensor = torch.zeros(n_samples, n_smile)

    d2_momo_tensor=torch.zeros(n_samples,n_momo)
    d2_protein_tensor = torch.zeros(n_samples, n_protein)
    d2_smile_tensor = torch.zeros(n_samples, n_smile)
    target_tensor=torch.zeros(n_samples,n_target)


    for i in range(n_samples):
        d1_momo_fts,d1_protein_fts,d1_fingerprint,d2_momo_fts,d2_protein_fts,d2_fingerprint,target = data[i]

        d1_momo_tensor[i]=torch.tensor(d1_momo_fts)
        d1_protein_tensor[i] = torch.tensor(d1_protein_fts)
        d1_smile_tensor[i] = torch.tensor(d1_fingerprint)

        d2_momo_tensor[i] = torch.tensor(d2_momo_fts)
        d2_protein_tensor[i] = torch.tensor(d2_protein_fts)
        d2_smile_tensor[i] = torch.tensor(d2_fingerprint)

        target_tensor[i]=torch.tensor(target)

    return d1_smile_tensor, d2_smile_tensor, target_tensor
    #return d1_momo_tensor,d1_protein_tensor,d1_smile_tensor,d2_momo_tensor,d2_protein_tensor,d2_smile_tensor,target_tensor





class MolGraphDataset(data.Dataset):
    #①由文件获得smiles list  逐个获取所有分子的邻接矩阵、原子特征、边特征
    r"""For datasets consisting of SMILES strings and target values.

    Expects a csv file formatted as:
    comment,smiles,targetName1,targetName2
    Some Comment,CN=C=O,0,1
    ,CC(=O)NCCC1=CNc2c1cc(OC)cc2,1,1

    Args:
        path
        prediction: set to True if dataset contains no target values
    """

    def __init__(self, drug_combo_file, prediction=False):
        print(drug_combo_file)
        drug_smile_file = './data/node_fts/drug_smiles.csv'
        drug_momo_file = './data/node_fts/drug_momo_fts.csv'
        drug_protein_file = './data/node_fts/drug_protein_fts.csv'

        #pairs
        self.drug1 = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[0], dtype=np.float, comments=None)
        self.drug2 = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[1], dtype=np.float, comments=None)
        #sider_effect
        self.side_effect=np.genfromtxt(drug_combo_file, delimiter=',', usecols=[2], dtype=np.float, comments=None)

        #fts
        self.drug_smile_fts = np.genfromtxt(drug_smile_file, skip_header=1, delimiter=',', usecols=[3],
                                        dtype=np.str_, comments=None)
        self.drug_momo_fts = np.genfromtxt(drug_momo_file, delimiter=',', usecols=range(1, 9263), dtype=np.float,
                                           comments=None)
        self.drug_protein_fts = np.genfromtxt(drug_protein_file, delimiter=',', usecols=range(1, 1034), dtype=np.float,
                                              comments=None)

        ###target
        # if prediction:
        #     self.target=self.targets = np.empty((len(self.drug1),1))
        # else:
        #     self.target = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[3], dtype=np.float,
        #                             comments=None)
        self.target = np.genfromtxt(drug_combo_file, delimiter=',', usecols=[3], dtype=np.float,comments=None)
        print(len(self.drug1))


    def __getitem__(self, index):
        d1_info = int(self.drug1[index])
        d2_info = int(self.drug2[index])

        adjacency_1, nodes_1, edges_1 = smile_to_graph(self.drug_smile_fts[d1_info])
        adjacency_2, nodes_2, edges_2 = smile_to_graph(self.drug_smile_fts[d2_info])

        d1_momo_fts = self.drug_momo_fts[d1_info]
        d1_protein_fts = self.drug_protein_fts[d1_info]
        d2_momo_fts = self.drug_momo_fts[d2_info]
        d2_protein_fts = self.drug_protein_fts[d2_info]

        #邻接矩阵 原子特征集合 边特征集

        d1_emb = smile_to_fingerprint(self.drug_smile_fts[d1_info])
        d2_emb = smile_to_fingerprint(self.drug_smile_fts[d2_info])

        side_effect=self.side_effect[index]
        targets = self.target[index]


        return  (adjacency_1, nodes_1, edges_1),(adjacency_2, nodes_2, edges_2),d1_momo_fts,d1_protein_fts,d2_momo_fts,d2_protein_fts,targets,side_effect,d1_emb,d2_emb

    def __len__(self):
        return len(self.drug1)


def smile_to_fingerprint(smile):
#获取分子的原子特征的函数
    #print(smile)
    molecule = Chem.MolFromSmiles(smile)
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(molecule,2,nBits=512)

    return fp1_morgan_hashed

def smile_to_graph(smile):
#获取分子的原子特征的函数
    #print(smile)
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()#节点数
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]#获得一个分子的所有原子
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)#获得这个分子的邻接矩阵
    node_features = np.array([atom_features(atom) for atom in atoms])#获得每个原子的特征

#获取边的特征
    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    #一个三维矩阵
    #起始节点，终止节点，边特征
    for bond in molecule.GetBonds():#获取每条边的特征
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return adjacency, node_features, edge_features
    #返回邻接矩阵，每个原子的特征（包含一系列化学性质），每条边的特征（边的特征其实就是边的类型）

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

def molgraph_collate_fn(data):
    #(adjacency_1, nodes_1, edges_1),(adjacency_2, nodes_2, edges_2),targets
    #把每一对数据的信息转化池化层tensor
    #把每一个drug的node_fts、edge_fts、adjacency、target都变成了tensor
    #将数据填充到该batch的最大长度
    n_samples = len(data)
    #print('len(data)',len(data))
    (adjacency_1, node_features_1, edge_features_1),(adjacency_2, node_features_2, edge_features_2),\
                    d1_momo_fts,d1_protein_fts,d2_momo_fts,d2_protein_fts,targets_0,side_effect,d1_emb,d_2emb = data[0]
    #分别获取d1和d2中院子最多的药物
    n_nodes_largest_graph_1 = max(map(lambda sample: sample[0][0].shape[0], data))
    n_nodes_largest_graph_2 = max(map(lambda sample: sample[1][0].shape[0], data))
    #print('n_nodes_largest_graph:',n_nodes_largest_graph)
    #找到最大的邻接矩阵

    n_node_features_1 = node_features_1.shape[1]
    n_edge_features_1 = edge_features_1.shape[2]
    n_node_features_2 = node_features_2.shape[1]
    n_edge_features_2 = edge_features_2.shape[2]
    n_targets = 1
    n_emb = len(d1_emb)
    n_momo = len(d1_momo_fts)
    n_protein = len(d1_protein_fts)
    n_side_effect=1

    adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
    node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_features_1)
    edge_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1, n_edge_features_1)
    d1_momo_tensor = torch.zeros(n_samples, n_momo)
    d1_protein_tensor = torch.zeros(n_samples, n_protein)


    adjacency_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2)
    node_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_node_features_2)
    edge_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2, n_edge_features_2)
    d2_momo_tensor = torch.zeros(n_samples, n_momo)
    d2_protein_tensor = torch.zeros(n_samples, n_protein)

    side_effect_tensor=torch.zeros(n_samples,n_side_effect)
    target_tensor = torch.zeros(n_samples, n_targets)

    d1_emb_tensor = torch.zeros(n_samples, n_emb)
    d2_emb_tensor = torch.zeros(n_samples, n_emb)

    for i in range(n_samples):
        (adjacency_1, node_features_1, edge_features_1), (adjacency_2, node_features_2, edge_features_2), \
        d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts, target,side_effect, d1_emb, d2_emb = data[i]

        n_nodes_1 = adjacency_1.shape[0]
        n_nodes_2 = adjacency_2.shape[0]


        adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(adjacency_1)
        node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(node_features_1)
        edge_tensor_1[i, :n_nodes_1, :n_nodes_1, :] = torch.Tensor(edge_features_1)
        d1_momo_tensor[i]=torch.Tensor(d1_momo_fts)
        d1_protein_tensor[i]=torch.Tensor(d1_protein_fts)

        adjacency_tensor_2[i, :n_nodes_2, :n_nodes_2] = torch.Tensor(adjacency_2)
        node_tensor_2[i, :n_nodes_2, :] = torch.Tensor(node_features_2)
        edge_tensor_2[i, :n_nodes_2, :n_nodes_2, :] = torch.Tensor(edge_features_2)
        d2_momo_tensor[i] = torch.Tensor(d2_momo_fts)
        d2_protein_tensor[i] = torch.Tensor(d2_protein_fts)

        side_effect_tensor[i]=torch.tensor(side_effect)
        target_tensor[i]=torch.tensor(target)

        d1_emb_tensor[i] = torch.IntTensor(d1_emb)
        d2_emb_tensor[i] = torch.IntTensor(d2_emb)


    return adjacency_tensor_1, node_tensor_1, edge_tensor_1, adjacency_tensor_2, node_tensor_2, edge_tensor_2,\
           d1_momo_tensor,d1_protein_tensor,d2_momo_tensor,d2_protein_tensor,target_tensor,side_effect_tensor,d1_emb_tensor,d2_emb_tensor
