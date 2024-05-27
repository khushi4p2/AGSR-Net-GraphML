import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from initializations import *
from preprocessing import normalize_adj_torch


class GSRLayer(nn.Module):

    ################## Chebyshev polynomials implementation
    # def __init__(self, hr_dim):
    def __init__(self, hr_dim, k):
        super(GSRLayer, self).__init__()

        ########################## CB poly imp.
        self.k = k
        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim, 2*hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)
        #######################################
        # self.weights = torch.from_numpy(
        #     weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        # self.weights = torch.nn.Parameter(
        #     data=self.weights, requires_grad=True)
    ################## CB. Poly Implementation

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):

            lr = A
            lr_dim = lr.shape[0]
            f = X

            ############################## debug code 
            print("dim of lr_dim:", lr_dim)
            print("dim of hr_dim:", hr_dim)
            ##########################################

            # Compute the Chebyshev polynomial approximation
            T_k = self.chebyshev_polynomials(lr, self.k)

            s_d = torch.cat((torch.eye(lr_dim), torch.eye(lr_dim)), 0)

            a = torch.matmul(self.weights, s_d)
            f_d = torch.zeros_like(f)

            # Perform the Chebyshev approximation
            for i in range(self.k):
                ############## debug code
                print(a.shape)
                print(T_k[i].shape)
                print(f.shape)
                ##########################
                f_d += torch.matmul(a, torch.matmul(T_k[i], f.t()))
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            X = torch.mm(adj, adj.t())
            X = (X + X.t())/2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)

    @staticmethod
    def chebyshev_polynomials(lr, k):
        """
        Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices.
        """
        T_k = [torch.eye(lr.shape[0]), lr.clone()]

        def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, lr):
            s = 2 * torch.matmul(lr, T_k_minus_one) - T_k_minus_two
            return s

        for i in range(2, k):
            T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], lr))

        return T_k

    # def forward(self, A, X):
    #     with torch.autograd.set_detect_anomaly(True):

    #         lr = A
    #         lr_dim = lr.shape[0]
    #         f = X

    #     ########################## CB. Poly Implementation
    #         # ############# fixed code - changed symeig function
    #         # eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')
    #         # ###############################

    #         # # U_lr = torch.abs(U_lr)
    #         # eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
    #         # s_d = torch.cat((eye_mat, eye_mat), 0)

    #         # a = torch.matmul(self.weights, s_d)
    #         # b = torch.matmul(a, torch.t(U_lr))
    #         # f_d = torch.matmul(b, f)
    #         # f_d = torch.abs(f_d)
    #         # f_d = f_d.fill_diagonal_(1)
    #         # adj = f_d

    #         # X = torch.mm(adj, adj.t())
    #         # X = (X + X.t())/2
    #         # X = X.fill_diagonal_(1)

    #         # Compute the normalized Laplacian matrix
    #         D = torch.sum(lr, dim=1)
    #         D_inv_sqrt = torch.pow(D, -0.5)
    #         D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    #         D_inv_sqrt = torch.diag(D_inv_sqrt)
    #         L = torch.eye(lr_dim) - torch.mm(D_inv_sqrt, lr).mm(D_inv_sqrt)
            
    #         # Compute Chebyshev polynomials
    #         T_k = [torch.eye(lr_dim), L]
    #         for k in range(2, self.k):
    #             T_k.append(2 * torch.mm(L, T_k[-1]) - T_k[-2])

    #         # Approximate the eigenvalue vector using Chebyshev polynomials
    #         eig_approx = torch.zeros(lr_dim, lr_dim)
    #         for k in range(self.k):
    #             eig_approx += torch.mm(self.weights[k][:lr_dim, :lr_dim], T_k[k])

    #         # Perform the graph super-resolution operation
    #         f_d = torch.mm(f, eig_approx)
    #         f_d = torch.abs(f_d)
    #         f_d = f_d.fill_diagonal_(1)
    #         adj = f_d

    #         X = torch.mm(adj, adj.t())
    #         X = (X + X.t()) / 2
    #         X = X.fill_diagonal_(1)

    #     return adj, torch.abs(X)

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, dropout, act=F.relu):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.act = act
#         self.weight = torch.nn.Parameter(
#             torch.FloatTensor(in_features, out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)

#     def forward(self, input, adj):
#         input = F.dropout(input, self.dropout, self.training)
#         support = torch.mm(input, self.weight)
#         output = torch.mm(adj, support)
#         output = self.act(output)
#         return output

############### Graph Attention Layer ################
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Define weights and a single-layer feedforward network for the attention mechanism
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Apply dropout to the input features
        input = F.dropout(input, self.dropout, self.training)
        # Linear transformation
        h = torch.mm(input, self.weight)

        # Prepare the attention mechanism
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1).view(N, N))

        # Only allow non-zero attention coefficients where adj is 1 (edges exist)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted sum of features produced by the attention coefficients
        h_prime = torch.matmul(attention, h)

        return h_prime

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.a)
