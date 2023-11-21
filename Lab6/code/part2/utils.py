"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import numpy as np
import torch


def sparse_mx_to_torch_sparse_tensor(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse_coo_tensor(indices, values, shape)