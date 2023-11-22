"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# @TODO: could use enum
READOUT_SUM, READOUT_MEAN = 'sum', 'mean'
MP_SUM, MP_MEAN = "sum", "mean"


class MessagePassing(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, neighbor_aggr: str):
        super(MessagePassing, self).__init__()
        self.neighbor_aggr = neighbor_aggr
        assert self.neighbor_aggr in [MP_SUM, MP_MEAN], "wrong aggregation mode"
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Message passing followed by readout (=pooling over graph)
        - with a "fully connected" fc1 projection for central nodes
        - with a different fc2 projection for neighbour nodes
        Warning:
        - There are no non linearities here.
        - Adjacency matrix is not expected to contain self loops (do no input A+I)
        Args:
            x (torch.Tensor): flattened batch input features [sum over N |v_i|, input_dim]
            adj (torch.Tensor): batched adjacency matrix without self loops

        Returns:
            torch.Tensor: N, ouput
        """
        # Task 6
        x_node = self.fc1(x)  # center
        x_nbrs = self.fc2(x)  # neighbors fc1!=fc2, x_node != x_nbrs
        m = torch.mm(adj, x_nbrs)  # message passing
        if self.neighbor_aggr == MP_SUM:
            output = x_node + m
        elif self.neighbor_aggr == MP_MEAN:
            deg = torch.spmm(adj, torch.ones(x.size(0), 1, device=x.device))
            output = x_node + torch.div(m, deg)
        else:
            raise KeyError(f"{self.neighbor_aggr} not supported")
        return output


class GNN(nn.Module):
    def __init__(
            self, input_dim: int, hidden_dim: int, output_dim: int, neighbor_aggr: str, readout: str, dropout: float):
        super(GNN, self).__init__()
        self.readout = readout
        assert neighbor_aggr in [MP_SUM, MP_MEAN], f"wrong neighborhood aggregation mode {neighbor_aggr}"
        assert readout in [READOUT_SUM, READOUT_MEAN], f"wrong readout mode {neighbor_aggr}"
        self.mp1 = MessagePassing(input_dim, hidden_dim, neighbor_aggr)
        self.mp2 = MessagePassing(hidden_dim, hidden_dim, neighbor_aggr)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor, idx: torch.Tensor):
        # Task 7
        x = self.relu(self.mp1(x, adj))
        x = self.dropout(x)
        x = self.relu(self.mp2(x, adj))
        x = self.dropout(x)

        if self.readout == READOUT_SUM:
            idx = idx.unsqueeze(1).repeat(1, x.size(1))
            out = torch.zeros(int(torch.max(idx))+1, x.size(1), device=x.device)
            out = out.scatter_add_(0, idx, x)
        elif self.readout == READOUT_MEAN:
            idx = idx.unsqueeze(1).repeat(1, x.size(1))
            out = torch.zeros(int(torch.max(idx))+1, x.size(1), device=x.device)
            out = out.scatter_add_(0, idx, x)
            count = torch.zeros(int(torch.max(idx))+1, x.size(1), device=x.device)
            count = count.scatter_add_(0, idx, torch.ones_like(x, device=x.device))
            out = torch.div(out, count)
        else:
            raise KeyError(f"wrong readout mode {self.readout}")

        # Task 7
        out = self.fc(out)
        return out
