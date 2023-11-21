"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)  # W0
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)  # W1
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in: torch.Tensor, adj_no_loops: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Apply layers

        Args:
            x_in (torch.Tensor): _description_
            adj_no_loops (torch.Tensor): Block matrix made of each adjacency matrix
            idx (torch.Tensor): [N*Vi] indexes for the batch
            [
                0
                0
                1
                1
                1
                2
                2
                3
                3
                3
                3
            ]

        Returns:
            torch.Tensor: [N] logits after softmax
        """

        # adj = adj_no_loops+torch.eye(x_in.shape[0], device=x_in.device)
        adj = adj_no_loops
        # 2 GCN layers
        z0 = self.relu(torch.mm(adj, self.fc1(x_in)))
        z1 = self.relu(torch.mm(adj, self.fc2(z0)))
        
        # READOUT
        idx = idx.unsqueeze(1).repeat(1, z1.size(1))
        out = torch.zeros(int(torch.max(idx)+1), z1.size(1), device=x_in.device)
        out = out.scatter_add_(0, idx, z1)  # readout

        # Classification layers
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return F.log_softmax(out, dim=1)
