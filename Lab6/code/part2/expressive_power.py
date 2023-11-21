"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'

# Task 4
# Sample 10 cycle graphs of size n= 10, ..., 19
dataset = [nx.cycle_graph(n) for n in range(10, 20)]


# Task 5
adj_matrices = [nx.adjacency_matrix(gr) for gr in dataset]
adj_block_diag = sp.block_diag(adj_matrices)

x = np.ones(adj_block_diag.shape[0])  # Features
idx = [np.ones(gr.number_of_nodes(), dtype=np.int32)*id for id, gr in enumerate(dataset)]
idx = np.concatenate(idx)

idx = torch.LongTensor(idx).to(device)
x = torch.FloatTensor(x).to(device)
adj_block_diag = sparse_mx_to_torch_sparse_tensor(adj_block_diag).to(device)
print(idx)
# print(x.shape)


# Task 8
model = GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
print(model(x, adj_block_diag, idx))

# Task 9

##################
# your code here #
##################


# Task 10

##################
# your code here #
##################


# Task 11

##################
# your code here #
##################
