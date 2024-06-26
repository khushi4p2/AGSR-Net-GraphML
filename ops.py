import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from preprocessing import calc_eigenvector_centrality


################## Attention Unpooling and Pooling ##################
class GraphAttentionUnpool(nn.Module):
    def __init__(self):
        super(GraphAttentionUnpool, self).__init__()
        self.attention = nn.Linear(320, 320)  # Added attention mechanism

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        attention_weights = torch.sigmoid(self.attention(X))  # Computing attention weights
        new_X[idx] = X * attention_weights  # Apply attention weights during unpooling
        return A, new_X

class GraphAttentionPool(nn.Module):
    def __init__(self, k, in_dim):
        super(GraphAttentionPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k * num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)  # attention scores applied to the node features
        A = A[idx, :]
        A = A[:, idx]  
        return A, new_X, idx

# modified for GAT Layer for pooling/unpooling + edge centrality measure
class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks

        self.start_gat = GraphAttentionLayer(in_dim + 1, dim, dropout=0.6, alpha=0.2, concat=True)
        self.bottom_gat = GraphAttentionLayer(dim, dim, dropout=0.6, alpha=0.2, concat=True)
        self.end_gat = GraphAttentionLayer(2*dim, out_dim, dropout=0.6, alpha=0.2, concat=False)

        self.down_gats = []
        self.up_gats = []
        
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gats.append(GraphAttentionLayer(dim, dim, dropout=0.6, alpha=0.2, concat=True))
            self.up_gats.append(GraphAttentionLayer(dim, dim, dropout=0.6, alpha=0.2, concat=True))
            self.pools.append(GraphAttentionPool(ks[i], dim))
            self.unpools.append(GraphAttentionUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []

        eigenvector_centrality = calc_eigenvector_centrality(A)
        X_with_centrality = torch.cat([X, eigenvector_centrality], dim=-1)

        X = self.start_gat(X_with_centrality, A)
        start_gat_outs = X

        org_X = X
        for i in range(self.l_n):
            X = self.down_gats[i](X, A)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)

        X = self.bottom_gat(X, A)
        
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)

            X = self.up_gats[i](X, A)

            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)

        X = self.end_gat(X, A)
        
        return X, start_gat_outs
        
# GAT Layers for pooling/unpooling in UNetBlock
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


