import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from math import floor

import util


class RGGE(nn.Module):
    def __init__(self, A, B, alpha, beta, eta, anc, disease_feat,
                 add_self_loop, drug_structure, target_seq,
                 feature_initial_type, feature_inital_params0, feature_inital_params1,
                 feature_learning_type, feature_learning_params,
                 distmult_loss_type, distmult_params,
                 graph_learning_type, graph_learning_params,
                 topological_heuristic_type, topological_heuristic_params,
                 ):
        super(RGGE, self).__init__()
        self.register_buffer('A', A.to(torch.float32), persistent=False)
        self.register_buffer('B', B.to(torch.float32), persistent=False)
        self.register_buffer('anc', anc, persistent=False)
        self.register_buffer('disease_feat', disease_feat, persistent=False)  # 这些变量会随着model.to(device)放入device
        self.register_buffer('drug_structure', torch.from_numpy(drug_structure), persistent=False)
        self.register_buffer('target_seq', torch.from_numpy(target_seq), persistent=False)
        # Hyperparameters.
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.add_self_loop = add_self_loop
        # Get initial feature
        self.gain_initial_feature0 = {
            'cnn': SeqEmbedding,  # lr=0.001
        }[feature_initial_type](**feature_inital_params0)
        self.gain_initial_feature1 = {
            'cnn': SeqEmbedding,
        }[feature_initial_type](**feature_inital_params1)
        # feature_learning
        self.feature_learning = {
            'rgcn': RGCNLinkPrediction,
        }[feature_learning_type](**feature_learning_params)
        self.distmult = {
            'Distmult': Distmult,
        }[distmult_loss_type](**distmult_params)
        self.graph_learning0 = {
            'mlp': PairwiseMLP,
        }[graph_learning_type](**graph_learning_params)
        self.graph_learning1 = {
            'mlp': PairwiseMLP,
        }[graph_learning_type](**graph_learning_params)
        self.topological_heuristic = {
            'ac': Autocovariance,
        }[topological_heuristic_type](**topological_heuristic_params)

        self.offset = [708, 708+1512, 708+1512+5603]

    def forward(self, DG_pos, DE_pos):
        # Positive masking.
        edges_pos = torch.vstack([DG_pos, DE_pos])
        B = self.B.index_put(tuple(edges_pos.t()),
                             torch.zeros(edges_pos.shape[0], device=self.A.device)) if self.training else self.B
        fledge = B.nonzero().to(self.A.device)
        type_matrix = util.get_adj_matrix().to(self.A.device)
        # 为rgcn准备特征
        drug_feature = self.gain_initial_feature0(self.drug_structure)
        target_feature = self.gain_initial_feature1(self.target_seq)
        # 拼接三个结点的特征得到X
        init_X = torch.cat((drug_feature, target_feature, self.disease_feat), dim=0)
        # 为rgcn准备边
        entity = self.feature_learning(init_X, fledge.T, type_matrix[fledge[:, 0], fledge[:, 1]])
        x_norm = F.normalize(entity.detach(), p=2, dim=1)
        S = torch.mm(x_norm, x_norm.t())

        if self.eta != 0.0:
            num_untrained_similarity_edges = floor((self.A != 0).sum() * self.eta)
            # 对角线设为0
            S.fill_diagonal_(0)
            S[:self.offset[0], :self.offset[0]] = 0
            S[self.offset[0]:, self.offset[0]:] = 0
            threshold = torch.kthvalue(S.view(-1), S.numel() - num_untrained_similarity_edges).values
            untrained_similarity_edge_mask = (S > threshold).bool()
        else:
            untrained_similarity_edge_mask = torch.zeros_like(S, dtype=torch.bool)
        augmented_edge_mask = self.A.to(bool) + untrained_similarity_edge_mask
        S = torch.relu(S * augmented_edge_mask)
        augmented_edges = augmented_edge_mask.triu().nonzero(as_tuple=False)
        # print(augmented_edge_mask.shape)
        ndg = augmented_edges[(augmented_edges[:, 0] < self.offset[0]) & (augmented_edges[:, 1] > self.offset[0]) & (augmented_edges[:, 1] < self.offset[1])]
        nde = augmented_edges[(augmented_edges[:, 0] < self.offset[0]) & (augmented_edges[:, 1] > self.offset[1]) & (augmented_edges[:, 1] < self.offset[2])]
        # print(ndg.shape, nde.shape)
        W = torch.zeros((self.A.shape[0], self.A.shape[0]), device=self.A.device)
        W[tuple(ndg.t())] = self.graph_learning0(entity, ndg)
        W[tuple(nde.t())] = self.graph_learning1(entity, nde)
        W = W + W.t()
        W.fill_diagonal_(1)
        # Positive masking.
        A = self.A.index_put(tuple(edges_pos.t()),
                             torch.zeros(edges_pos.shape[0], device=self.A.device)) if self.training else self.A
        # Combine topological edge weights, trained edge weights, and untrained edge weights.
        A_enhanced = self.alpha * A + (1 - self.alpha) * (
                (A.to(bool) + untrained_similarity_edge_mask) *
                (self.beta * W + (1 - self.beta) * S)
        )
        if self.add_self_loop:
            A_enhanced.fill_diagonal_(1)  # Add self-loop to all nodes.
        else:
            A_enhanced.diagonal()[A_enhanced.sum(dim=1) == 0] = 1.0  # Add self-loops to isolated nodes.

        R = self.topological_heuristic(A_enhanced)

        return entity, R


class SeqEmbedding(nn.Module):
    def __init__(self, dict_len, embedding_dim, num_filters, filter_length, dropout_rate=0.5, output_dim=128):
        super(SeqEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_len + 1, embedding_dim, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=filter_length),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=filter_length),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=filter_length),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters * 3, output_dim)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = self.convs(x)  # 卷积层序列化
        x = self.pool(x).squeeze(-1)  # 池化
        x = self.fc(x)  # 全连接层
        return x


class RGCNLinkPrediction(nn.Module):
    def __init__(self,  n_relations, n_bases, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(RGCNLinkPrediction, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, n_relations, num_bases=n_bases))
        self.bns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, n_relations, num_bases=n_bases))
            self.bns.append(nn.LayerNorm(hidden_channels))
        if num_layers > 1:
            self.convs.append(RGCNConv(hidden_channels, out_channels, n_relations, num_bases=n_bases))
            self.bns.append(nn.LayerNorm(out_channels))
        elif num_layers == 1:
            self.convs[0] = RGCNConv(in_channels, out_channels, n_relations, num_bases=n_bases)
            self.bns[0] = nn.LayerNorm(out_channels)

        self.activation = nn.LeakyReLU(0.1)

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)

    def forward(self, entity, train_pos_edge_index, train_pos_edge_types):
        x = entity
        for i in range(self.num_layers):
            x = self.convs[i](x, train_pos_edge_index, train_pos_edge_types)
            x = self.bns[i](x)
            x = self.activation(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class Distmult(nn.Module):
    def __init__(self, n_relations, out_channels):
        super(Distmult, self).__init__()
        self.relation_embedding = nn.Parameter(torch.Tensor(n_relations, out_channels))
        nn.init.xavier_uniform_(self.relation_embedding, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, entity, edge, type):
        s = entity[edge[:, 0]]
        r = self.relation_embedding[type]
        o = entity[edge[:, 1]]
        return torch.sum(s * r * o, dim=1)

    def reg_loss(self, entity):
        return torch.mean(entity.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, relu_first, batch_norm):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if batch_norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first
        self.batch_norm = batch_norm

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            if self.batch_norm:
                x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)

            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class PairwiseMLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, relu_first, batch_norm, permutation_invariant):

        super(PairwiseMLP, self).__init__()
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers, dropout, relu_first, batch_norm)
        self.permutation_invariant = permutation_invariant

    def forward(self, x, edges):

        if self.permutation_invariant:
            edge_x = torch.cat([x[edges[:, 0]] + x[edges[:, 1]], torch.abs(x[edges[:, 0]] - x[edges[:, 1]])], dim=1)
        else:
            edge_x = torch.cat([x[edges[:, 0]], x[edges[:, 1]]], dim=1)

        return self.mlp(edge_x).exp()[:, 1]


class Autocovariance(torch.nn.Module):

    def __init__(self, scaling_parameter):
        super(Autocovariance, self).__init__()
        self.scaling_parameter = scaling_parameter

    def forward(self, A):

        # Compute Autocovariance matrix.
        d = A.sum(dim=1)
        pi = F.normalize(d, p=1, dim=0)
        M = A / d[:, None]
        R = torch.diag(pi) @ torch.matrix_power(M, self.scaling_parameter) - torch.outer(pi, pi)

        # Standardize Autocovariance entries.
        R = (R - R.mean())/R.std()
        return R