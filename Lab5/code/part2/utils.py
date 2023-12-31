"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import scipy.sparse as sp
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
############## Task 9
def normalize_adjacency(A_no_self_loop: sp.sparray, add_self_loops=True) -> sp.sparray:
    """Compute normalized adjacency matrix  D^-1/2 . A . D^-1/2

    Args:
        A_no_self_loop (sp.sparray): Adjacency matrix (extracted from networkX for instance)

    Returns:
        sp.sparray: Normalized adjacency matrix
    """
    if add_self_loops:
        A = A_no_self_loop + sp.eye(A_no_self_loop.shape[0])
    else:
        A = A_no_self_loop
    D = A.sum(axis=1) # degree is the sum over columns
    D = np.squeeze(np.asarray(D))
    # Compute D^-1/2
    inv_d = 1./np.sqrt(D)
    
    D_inv = sp.diags(inv_d)
    A_normalized = D_inv @ A @ D_inv
    # Compute D^-1/2 . A  . D^-1/2
    return A_normalized



def load_cora(root_dir=Path(__file__).parent/".."/"data"):
    idx_features_labels = np.genfromtxt(root_dir/"cora.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = features.todense()
    features /= features.sum(1).reshape(-1, 1)
    
    class_labels = idx_features_labels[:, -1]
    le = LabelEncoder()
    class_labels = le.fit_transform(class_labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(root_dir/"cora.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(class_labels.size, class_labels.size), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, class_labels


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
