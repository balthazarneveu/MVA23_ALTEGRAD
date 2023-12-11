"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional


def normalize_adjacency(A):
    # Normalizes adjacency matrix represnted as a sparse SciPy matrix
    n = A.shape[0]
    A += sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D_inv = sp.diags(inv_degs)
    A_normalized = D_inv.dot(A)
    return A_normalized


def sparse_mx_to_torch_sparse(M):
    # Converts a sparse SciPy matrix to a sparse PyTorch tensor
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def find_communities_and_plot(G, save_path: Optional[Path] = None, fig_name=""):
    # Compute the best partition using Louvain algorithm
    partition = nx.community.louvain_communities(G)

    # Reorder the adjacency matrix
    ordered_nodes = list()
    for p in partition:
        for node in p:
            ordered_nodes.append(node)

    reordered_matrix = nx.to_numpy_array(G, nodelist=ordered_nodes)

    # Plot adjacency matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(reordered_matrix, cmap='gray')
    plt.title("Reordered Adjacency Matrix")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path/f"{fig_name}_adjacency_matrix.png")
    plt.close()
    # Draw the graph
    plt.figure(figsize=(5, 5))
    plt.title('Generated graph')
    nx.draw(G)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path/f"{fig_name}_generated_graph.png")
    plt.close()