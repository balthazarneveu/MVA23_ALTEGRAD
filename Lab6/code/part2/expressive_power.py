"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN, MP_SUM, MP_MEAN, READOUT_SUM, READOUT_MEAN
from utils import sparse_mx_to_torch_sparse_tensor
from itertools import product

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
# Concatenating all graphs into a big batched matrix
# will result in 10+11+12+..+19 = 145 nodes where there are 10 connected components
# Batch Adjacency matrix will be of size 145x145

# Task 5
adj_matrices = [nx.adjacency_matrix(gr) for gr in dataset]
adj_block_diag = sp.block_diag(adj_matrices)
idx = [np.ones(gr.number_of_nodes(), dtype=np.int32)*id for id, gr in enumerate(dataset)]
idx = np.concatenate(idx)
idx = torch.LongTensor(idx).to(device)

x = np.ones((adj_block_diag.shape[0], 1))  # Features
x = torch.FloatTensor(x).to(device)
adj_block_diag = sparse_mx_to_torch_sparse_tensor(adj_block_diag).to(device)


# Task 8
input_dim = 1
for  neighbor_aggr, readout in product([MP_MEAN, MP_SUM], [READOUT_MEAN, READOUT_SUM]):
    with torch.no_grad():
        model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
        out_representation = model(x, adj_block_diag, idx)
    print(f"{neighbor_aggr=}, {readout=} \n {out_representation.detach().cpu().numpy()}") 

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
