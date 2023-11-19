"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Tasks 10 and 13
class GNN(nn.Module):
    """Simple GNN model
    """
    def __init__(self, n_feat: int, n_hidden_1: int, n_hidden_2: int, n_class: int, dropout: float,
                 return_hidden_features: bool = False):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.return_hidden_features = return_hidden_features

    def forward(self, x_in, adj):
        z0 = self.fc1(x_in)
        z0 = self.relu(torch.mm(adj, z0))
        z0 = self.dropout(z0)
        # remark on dropout: we drop some of the tensor components
        # not the weights
        # This is equivalent to dropping full rows in the weight matrix
        # not dropping random weights

        z1 = self.fc2(z0)
        hidden_feature = z1.clone()
        z1 = self.relu(torch.mm(adj, z1))

        x = self.fc3(z1)
        if self.return_hidden_features:
            return F.log_softmax(x, dim=1), hidden_feature
        else:
            return F.log_softmax(x, dim=1)
