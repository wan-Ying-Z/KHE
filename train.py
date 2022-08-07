import torch
from torch import optim
from torch.utils.data import DataLoader

import gnn.aggregation_mpnn_implementations

from losses import LOSS_FUNCTIONS
from train_logging import LOG_FUNCTIONS
from gnn.molgraph_data  import MolGraphDataset, molgraph_collate_fn

import argparse
import  numpy as np


MODEL_CONSTRUCTOR_DICTS = {
    'AttentionGGNN': {  # the below, batch size 50, learn rate 1.560e-5 and 600 epochs is good for BBBP
        'constructor': gnn.aggregation_mpnn_implementations.AttentionGGNN,  # 类实例构造器
        'hyperparameters': {
            'message-passes': {'type': int, 'default': 2},
            'message-size': {'type': int, 'default': 25},
            'msg-depth': {'type': int, 'default': 2},
            'msg-hidden-dim': {'type': int, 'default': 50},  # 来自相邻节点的消息大小
            'msg-dropout-p': {'type': float, 'default': 0.0},
            'att-depth': {'type': int, 'default': 2},  # attention层的CNN的层数
            'att-hidden-dim': {'type': int, 'default': 50},  # attention不改变输入输出的大小
            'att-dropout-p': {'type': float, 'default': 0.0},
            'gather-width': {'type': int, 'default': 45},  #
            'gather-att-depth': {'type': int, 'default': 2},
            'gather-att-hidden-dim': {'type': int, 'default': 45},
            'gather-att-dropout-p': {'type': float, 'default': 0.0},
            'gather-emb-depth': {'type': int, 'default': 2},
            'gather-emb-hidden-dim': {'type': int, 'default': 26},
            'gather-emb-dropout-p': {'type': float, 'default': 0.0},
            'out-depth': {'type': int, 'default': 2},  # 全连接网络的深度
            'out-hidden-dim': {'type': int, 'default': 90},  # 全连接网络的输出向量大小
            'out-dropout-p': {'type': float, 'default': 0.1},
            'out-layer-shrinkage': {'type': float, 'default': 0.6}
        }
    }
}



common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

common_args_parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')
common_args_parser.add_argument('--train-set', type=str, default='./data/model/train.csv', help='Training dataset path')
common_args_parser.add_argument('--test-set', type=str, default='./data/model/test.csv', help='Training dataset path')
common_args_parser.add_argument('--valid-set', type=str, default='./data/model/validation.csv', help='Training dataset path')

common_args_parser.add_argument('--loss', type=str, default='CrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='All', help='roc-auc or MSE or All')

common_args_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
common_args_parser.add_argument('--batch-size', type=int, default=128, help='Number of graphs in a mini-batch')
common_args_parser.add_argument('--learn-rate', type=float, default=1e-4)

common_args_parser.add_argument('--savemodel', action='store_true', default=True, help='Saves model with highest validation score')
common_args_parser.add_argument('--logging', type=str, default='less', choices=[k for k, v in LOG_FUNCTIONS.items()])


main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = main_parser.add_subparsers(help=', '.join([k for k, v in MODEL_CONSTRUCTOR_DICTS.items()]), dest='model')
subparsers.required = True

model_parsers = {}
for model_name, constructor_dict in MODEL_CONSTRUCTOR_DICTS.items():
    subparser = subparsers.add_parser(model_name, parents=[common_args_parser])
    for hp_name, hp_kwargs in constructor_dict['hyperparameters'].items():
        subparser.add_argument('--' + hp_name, **hp_kwargs, help=model_name + ' hyperparameter')
    model_parsers[model_name] = subparser


def main():
    global args
    args = main_parser.parse_args()#将main中实例的所有属性赋予args
    args_dict = vars(args)#变量赋值
    # dictionary of hyperparameters that are specific to the chosen model  特定于所选模型的超参数字典
    model_hp_kwargs = {
        name.replace('-', '_'): args_dict[name.replace('-', '_')]   # argparse converts to "_" implicitly
        for name, v in MODEL_CONSTRUCTOR_DICTS[args.model]['hyperparameters'].items()
    }
    train_dataset = MolGraphDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=molgraph_collate_fn)

    validation_dataset = MolGraphDataset(args.valid_set)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=molgraph_collate_fn)

    test_dataset = MolGraphDataset(args.test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=molgraph_collate_fn)

    ((sample_adjacency_1, sample_nodes_1, sample_edges_1),(sample_adjacency_2, sample_nodes_2, sample_edges_2),d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts,sample_target,side_effect,d1,d2)= train_dataset[0]


    #调用模型
    #给模型传入了输入特原子特征长度、边特征长度、输出特征长度
    net = MODEL_CONSTRUCTOR_DICTS[args.model]['constructor'](
#        node_features=len(np.array(sample_nodes[0])), edge_features=len(np.array(sample_edges[0, 0])), out_features=len(np.array(sample_target)),
        node_features_1=len(np.array(sample_nodes_1[0])), edge_features_1=len(np.array(sample_edges_1[0, 0])),
        node_features_2=len(np.array(sample_nodes_2[0])), edge_features_2=len(np.array(sample_edges_2[0, 0])),
        out_features=1
        ,**model_hp_kwargs)

    if args.cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.learn_rate)#优化器

    criterion = LOSS_FUNCTIONS[args.loss]#损失函数

    for epoch in range(args.epochs):
        net.train()

        for i_batch, batch in enumerate(train_dataloader):

            if args.cuda:
                batch = [tensor.cuda() for tensor in batch]

            adjacency_1, nodes_1, edges_1,adjacency_2, nodes_2, edges_2,d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts,\
                                                                         target,side_effect,d1,d2 = batch

            optimizer.zero_grad()

            output = net(adjacency_1, nodes_1, edges_1,adjacency_2, nodes_2, edges_2,d1_momo_fts, d1_protein_fts, d2_momo_fts, d2_protein_fts,side_effect)
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)#将参数归一化

            optimizer.step()


        with torch.no_grad():
            net.eval()
            LOG_FUNCTIONS[args.logging](
                net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args
            )


if __name__ == '__main__':
    main()
