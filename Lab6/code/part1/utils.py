"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import torch
from random import randint
from typing import Tuple, List


# Task 1
def create_dataset(
    num_graph: int = 50,
    probabilities: List[float] = [0.2, 0.4]
) -> Tuple[List[nx.Graph], List[int]]:
    """Samples a dataset of Erdos-Rényi random graphs with various probabilities (corresponding to classes).
    Args:
        num_graph (int, optional): amount of graphs to sample in each class. Defaults to 50.
        probabilities (List[float], optional): Probability p of connecting 2 nodes.
        Defaults to [0.2, 0.4].
    Returns:
        Tuple[List[nx.Graph], List[int]]: Graphs, Labels
        We are studying a classification task here:
        Label 0 is assigned to the graphs of proability 0.2
        Label 1 is assigned to the graphs of proability 0.4
    """
    graph_list = list()
    labels_list = list()
    for label, p in enumerate(probabilities):
        for _ in range(num_graph):
            n = randint(10, 20)
            graph_list.append(nx.fast_gnp_random_graph(n, p=p))
            labels_list.append(label)
    return graph_list, labels_list


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


if __name__ == "__main__":
    gs_list, y = create_dataset()
    print(gs_list[0])
