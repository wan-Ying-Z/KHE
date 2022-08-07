import torch
from torch import nn




class AggregationMPNN(nn.Module):

    def __init__(self, node_features_1,edge_features_1,node_features_2,edge_features_2, message_size, message_passes, out_features):

        super(AggregationMPNN, self).__init__()
        self.node_features_1 = node_features_1
        self.edge_features_1 = edge_features_1
        self.node_features_2 = node_features_2
        self.edge_features_2 = edge_features_2
        self.message_size = message_size
        self.message_passes = message_passes
        self.out_features = out_features

        self.proteinemb = Embeddings(1033, 50, 1033, 0)
        self.protein_encoder = nn.Sequential(
            # nn.RNN(input_size=self.emb_size, hidden_size=16, dropout=0.5)
            nn.LSTM(input_size=50, hidden_size=8, dropout=0.0)
        )

        self.protein_decoder = nn.Sequential(
            nn.Linear(8 * 1033, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 45),
        )

    # nodes (total number of nodes in batch, number of features)
    # node_neighbours (total number of nodes in batch, max node degree, number of features)
    # node_neighbours (total number of nodes in batch, max node degree, number of edge features)
    # mask (total number of nodes in batch, max node degree) elements are 1 if corresponding neighbour exist
    def aggregate_message(self, nodes, node_neighbours, edges, mask):

        raise NotImplementedError

    # inputs are "batches" of shape (maximum number of nodes in batch, number of features)
    def update_1(self, nodes, messages):

        raise NotImplementedError
    def update_2(self, nodes, messages):

        raise NotImplementedError

    # inputs are "batches" of same shape as the nodes passed to update
    # node_mask is same shape as inputs and is 1 if elements corresponding exists, otherwise 0
    def readout_1(self, hidden_nodes, input_nodes, node_mask):

        raise NotImplementedError
    def readout_2(self, hidden_nodes, input_nodes, node_mask):

        raise NotImplementedError
    def final_layer(self,out):

        raise NotImplementedError

    def side_encoder(self):
        raise NotImplementedError
    # def protein_encoder(self):
    #     raise NotImplementedError

    def momo_encoder(self):
        raise NotImplementedError



    def forward(self, adjacency_1, nodes_1, edges_1,adjacency_2, nodes_2, edges_2,d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts,side_effect):
        #print('奇怪的forward')
        #处理第一个smiles
        edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1 = adjacency_1.nonzero().unbind(-1)
        node_batch_batch_indices_1, node_batch_node_indices_1 = adjacency_1.sum(-1).nonzero().unbind(-1)
        node_batch_adj_1 = adjacency_1[node_batch_batch_indices_1, node_batch_node_indices_1, :]

        node_batch_size_1 = node_batch_batch_indices_1.shape[0]
        #node_degree表示每个节点的度数
        node_degrees_1 = node_batch_adj_1.sum(-1).long()
        max_node_degree_1 = node_degrees_1.max()
        node_batch_node_neighbours_1 = torch.zeros(node_batch_size_1, max_node_degree_1, self.node_features_1)
        node_batch_edges_1 = torch.zeros(node_batch_size_1, max_node_degree_1, self.edge_features_1)
        #每个原子都有一个max_node_degree_1*self.edge_features_1的原子邻接矩阵，并且标识连接的边是哪一种类

        node_batch_neighbour_neighbour_indices_1 = torch.cat([torch.arange(i) for i in node_degrees_1])

        edge_batch_node_batch_indices_1 = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees_1)]
        ).long()

        node_batch_node_neighbour_mask_1 = torch.zeros(node_batch_size_1, max_node_degree_1)


        #处理第二个smiles
        edge_batch_batch_indices_2, edge_batch_node_indices_2, edge_batch_neighbour_indices_2 = adjacency_2.nonzero().unbind(-1)

        node_batch_batch_indices_2, node_batch_node_indices_2 = adjacency_2.sum(-1).nonzero().unbind(-1)
        node_batch_adj_2 = adjacency_2[node_batch_batch_indices_2, node_batch_node_indices_2, :]

        node_batch_size_2 = node_batch_batch_indices_2.shape[0]
        node_degrees_2 = node_batch_adj_2.sum(-1).long()

        max_node_degree_2 = node_degrees_2.max()
        node_batch_node_neighbours_2 = torch.zeros(node_batch_size_2, max_node_degree_2, self.node_features_2)
        node_batch_edges_2 = torch.zeros(node_batch_size_2, max_node_degree_2, self.edge_features_2)

        node_batch_neighbour_neighbour_indices_2 = torch.cat([torch.arange(i) for i in node_degrees_2])

        edge_batch_node_batch_indices_2 = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees_2)]
        ).long()

        node_batch_node_neighbour_mask_2 = torch.zeros(node_batch_size_2, max_node_degree_2)

        ##########两个smiles都处理完了
        if next(self.parameters()).is_cuda:
            node_batch_node_neighbours_1 = node_batch_node_neighbours_1.cuda()
            node_batch_edges_1 = node_batch_edges_1.cuda()
            node_batch_neighbour_neighbour_indices_1 = node_batch_neighbour_neighbour_indices_1.cuda()
            edge_batch_node_batch_indices_1 = edge_batch_node_batch_indices_1.cuda()
            node_batch_node_neighbour_mask_1 = node_batch_node_neighbour_mask_1.cuda()

            node_batch_node_neighbours_2 = node_batch_node_neighbours_2.cuda()
            node_batch_edges_2 = node_batch_edges_2.cuda()
            node_batch_neighbour_neighbour_indices_2 = node_batch_neighbour_neighbour_indices_2.cuda()
            edge_batch_node_batch_indices_2 = edge_batch_node_batch_indices_2.cuda()
            node_batch_node_neighbour_mask_2 = node_batch_node_neighbour_mask_2.cuda()

            d1_momo_fts = d1_momo_fts.cuda()
            d1_protein_fts = d1_protein_fts.cuda()
            d2_momo_fts = d2_momo_fts.cuda()
            d2_protein_fts = d2_protein_fts.cuda()
            side_effect = side_effect.cuda()

        #处理第一个smiles
        #节点的邻居mask=1
        #node_batch_neighbour:n_nodes*max_node_degree
        node_batch_node_neighbour_mask_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1] = 1
        #print('edges_1:',edges_1.shape)
        #node_batch_edges_1:n_nodes*75
        node_batch_edges_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1, :] = \
            edges_1[edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1, :]
        hidden_nodes_1 = nodes_1.clone()


        #处理第二个smiles
        node_batch_node_neighbour_mask_2[edge_batch_node_batch_indices_2, node_batch_neighbour_neighbour_indices_2] = 1
        node_batch_edges_2[edge_batch_node_batch_indices_2, node_batch_neighbour_neighbour_indices_2, :] = \
            edges_2[edge_batch_batch_indices_2, edge_batch_node_indices_2, edge_batch_neighbour_indices_2, :]
        hidden_nodes_2 = nodes_2.clone()
        #处理完啦

        for i in range(self.message_passes):
            #处理第一个similes
            node_batch_nodes_1 = hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1, :]
            node_batch_node_neighbours_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1, :] = \
                hidden_nodes_1[edge_batch_batch_indices_1, edge_batch_neighbour_indices_1, :]

            messages_1 = self.aggregate_message_1(
                node_batch_nodes_1, node_batch_node_neighbours_1.clone(), node_batch_edges_1, node_batch_node_neighbour_mask_1
            )
            hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1, :] = self.update_1(node_batch_nodes_1, messages_1)

            # 处理第二个similes
            node_batch_nodes_2 = hidden_nodes_2[node_batch_batch_indices_2, node_batch_node_indices_2, :]
            node_batch_node_neighbours_2[edge_batch_node_batch_indices_2, node_batch_neighbour_neighbour_indices_2, :] = \
                hidden_nodes_2[edge_batch_batch_indices_2, edge_batch_neighbour_indices_2, :]

            messages_2 = self.aggregate_message_2(
                node_batch_nodes_2, node_batch_node_neighbours_2.clone(), node_batch_edges_2,
                node_batch_node_neighbour_mask_2
            )
            hidden_nodes_2[node_batch_batch_indices_2, node_batch_node_indices_2, :] = self.update_2(node_batch_nodes_2,
                                                                                                   messages_2)
            ######处理完啦



        node_mask_1 = (adjacency_1.sum(-1) != 0)
        output_1 = self.readout_1(hidden_nodes_1, nodes_1, node_mask_1)

        node_mask_2 = (adjacency_2.sum(-1) != 0)  # .unsqueeze(-1).expand_as(nodes)
        output_2 = self.readout_2(hidden_nodes_2, nodes_2, node_mask_2)

        #mpnn特征
        #output_drug=torch.cat((output_1,output_2),dim=1)

        #蛋白质特征
        # p1_fts=self.protein_encoder(d1_protein_fts)
        # p2_fts = self.protein_encoder(d2_protein_fts)
        batchsize=d1_protein_fts.size(0)

        p1 = self.proteinemb(d1_protein_fts)
        p2 = self.proteinemb(d2_protein_fts)

        p1_, _ = self.protein_encoder(p1)
        p2_, _ = self.protein_encoder(p2)
        p1_ = p1_.view(batchsize, -1)
        p2_ = p2_.view(batchsize, -1)

        p1_fts = self.protein_decoder(p1_)
        p2_fts = self.protein_decoder(p2_)


        #momo特征
        momo1_fts=self.momo_encoder(d1_momo_fts)
        momo2_fts = self.momo_encoder(d2_momo_fts)

        #特征混合过程
        o1=torch.cat((output_1,p1_fts),dim=1)
        drug1_fts=torch.cat((o1,momo1_fts),dim=1)

        o2 = torch.cat((output_2, p2_fts), dim=1)
        drug2_fts = torch.cat((o2, momo2_fts), dim=1)

        output_drug=torch.cat((drug1_fts,drug2_fts),dim=1)

        #副作用特征 90
        side_effect=self.side_encoder(side_effect)


        output=torch.cat((output_drug,side_effect),dim=1)
        #print(output.size())
        out=self.final_layer(output)

        return out



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b=torch.LongTensor(1,2)
        input_ids=input_ids.type_as(b)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)#【1.。。50】
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)#64*50
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings