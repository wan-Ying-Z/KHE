import torch
from torch import nn

from gnn.aggregation_mpnn import AggregationMPNN
from gnn.modules import FeedForwardNetwork, Set2Vec, GraphGather,Side_Encoder,Momo_Encoder,Protein_Encoder,Structure_Encoder


class AttentionENNS2V(AggregationMPNN):

    def __init__(self, node_features, edge_features, message_size, message_passes, out_features,
                 enn_depth=3, enn_hidden_dim=200, enn_dropout_p=0,
                 att_depth=3, att_hidden_dim=200, att_dropout_p=0,
                 s2v_lstm_computations=12, s2v_memory_size=50,
                 out_depth=1, out_hidden_dim=200, out_dropout_p=0):
        super(AttentionENNS2V, self).__init__(
            node_features, edge_features, message_size, message_passes, out_features
        )
        self.enn = FeedForwardNetwork(
            edge_features, [enn_hidden_dim] * enn_depth, node_features * message_size, dropout_p=enn_dropout_p
        )
        self.att_enn = FeedForwardNetwork(
            node_features + edge_features, [att_hidden_dim] * att_depth, message_size, dropout_p=att_dropout_p
        )
        self.gru = nn.GRUCell(input_size=message_size, hidden_size=node_features, bias=False)
        self.s2v = Set2Vec(node_features, s2v_lstm_computations, s2v_memory_size)
        self.out_nn = FeedForwardNetwork(
            s2v_memory_size * 2, [out_hidden_dim] * out_depth, out_features, dropout_p=out_dropout_p, bias=False
        )

    def aggregate_message(self, nodes, node_neighbours, edges, mask):
        BIG_NEGATIVE = -1e6
        max_node_degree = node_neighbours.shape[1]

        enn_output = self.enn(edges)
        matrices = enn_output.view(-1, max_node_degree, self.message_size, self.node_features)
        message_terms = torch.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze()

        att_enn_output = self.att_enn(torch.cat([edges, node_neighbours], dim=2))
        energies = att_enn_output.view(-1, max_node_degree, self.message_size)
        energy_mask = (1 - mask).float() * BIG_NEGATIVE
        weights = torch.softmax(energies + energy_mask.unsqueeze(-1), dim=1)

        return (weights * message_terms).sum(1)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)


class AttentionGGNN(AggregationMPNN):

    def __init__(self, node_features_1, edge_features_1, node_features_2, edge_features_2,message_size, message_passes, out_features,
                 msg_depth=4, msg_hidden_dim=200, msg_dropout_p=0.0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0.0, out_layer_shrinkage=1.0):
        super(AttentionGGNN, self).__init__(node_features_1, edge_features_1,node_features_2, edge_features_2, message_size, message_passes, out_features)

        self.msg_nns_1 = nn.ModuleList()

        self.msg_nns_2 = nn.ModuleList()
        for _ in range(edge_features_1):
            #对于每一条边
            self.msg_nns_1.append(
                FeedForwardNetwork(node_features_1, [msg_hidden_dim] * msg_depth, message_size, dropout_p=msg_dropout_p, bias=False)
            )

        for _ in range(edge_features_2):

            self.msg_nns_2.append(
                FeedForwardNetwork(node_features_2, [msg_hidden_dim] * msg_depth, message_size, dropout_p=msg_dropout_p, bias=False)
            )

        self.gru_1 = nn.GRUCell(input_size=message_size, hidden_size=node_features_1, bias=False)
        self.gru_2 = nn.GRUCell(input_size=message_size, hidden_size=node_features_2, bias=False)

        self.gather_1 = GraphGather(
            node_features_1, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        self.gather_2 = GraphGather(
            node_features_2, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )


        out_layer_sizes = [  # example: depth 5, dim 50, shrinkage 0.5 => out_layer_sizes [50, 42, 35, 30, 25]
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width*6+90, out_layer_sizes, out_features, dropout_p=out_dropout_p)

        self.side_effect_encoder=Side_Encoder()
        self.momoencoder=Momo_Encoder()
        self.struencoder = Structure_Encoder()

    def aggregate_message_1(self, nodes, node_neighbours, edges, node_neighbour_mask):

        energy_mask = (node_neighbour_mask == 0).float() * 1e6
        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns_1[i](node_neighbours) for i in range(self.edge_features_1)
        ]#给第i类边进行masked
        embedding = sum(embeddings_masked_per_edge)

        return torch.sum(embedding,dim=1)



    def aggregate_message_2(self, nodes, node_neighbours, edges, node_neighbour_mask):

        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns_2[i](node_neighbours) for i in range(self.edge_features_2)
        ]
        embedding = sum(embeddings_masked_per_edge)

        return torch.sum(embedding, dim=1)

    def update_1(self, nodes, messages):
        return self.gru_1(messages, nodes)

    def readout_1(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather_1(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings

    def update_2(self, nodes, messages):
        return self.gru_2(messages, nodes)

    def readout_2(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather_2(hidden_nodes, input_nodes, node_mask)
        return graph_embeddings

    def final_layer(self,connected_vector):
        embed=self.out_nn(connected_vector)

        return embed

    def side_encoder(self,input):
        side_effect=self.side_effect_encoder(input)
        return side_effect

    def momo_encoder(self,input):
        momo_fts=self.momoencoder(input)
        return momo_fts

    def stru_encoder(self, input):
        stru_fts = self.struencoder(input)
        return stru_fts