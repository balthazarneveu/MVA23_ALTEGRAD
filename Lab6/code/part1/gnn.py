"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from typing import Tuple, List
from models import GNN
from utils import create_dataset, sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 200
batch_size = 8
n_hidden_1 = 16
n_hidden_2 = 32
n_hidden_3 = 32
learning_rate = 0.01

# Generates synthetic dataset
Gs, y = create_dataset()
n_class = np.unique(y).size

# Splits the dataset into a training and a test set - Shuffle
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

N_train = len(G_train)
N_test = len(G_test)

# Initializes model and optimizer
model = GNN(1, n_hidden_1, n_hidden_2, n_hidden_3, n_class).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()


def get_batch(
    graph_list: List[nx.Graph],
    labels: List[int],
    start_index: int,
    end_index: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_graphs = graph_list[start_index:end_index]  # Batch of graph elements
    # Compute adjacency matrix of the batch of size N
    # = big a bigger graph with N connected components.
    adj_batch = [nx.adjacency_matrix(gr) for gr in current_graphs]
    adj_batch = sp.block_diag(adj_batch)
    adj_batch += sp.eye(adj_batch.shape[0]) # Add the self loop to the adjacency matrix.
    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)

    # Input features (stacked for all nodes of the batches of the graph)
    features_batch = torch.ones((adj_batch.shape[0], 1), dtype=torch.float).to(device)

    # Indexes used for the readout layer
    # Note: Used to retrieve which components of the big batched graph
    # correspond to which original batch component.
    n_nodes = [g.number_of_nodes() for g in current_graphs]  # Number of nodes in each element of the batch
    idx_batch = [idx*torch.ones(total_node, dtype=torch.long) for idx, total_node in enumerate(n_nodes)]
    idx_batch = torch.cat(idx_batch).to(device)

    # Labels
    labels_batch = torch.LongTensor(labels[start_index:end_index]).to(device)
    return features_batch, adj_batch, idx_batch, labels_batch


# Trains the model
for epoch in range(epochs):
    t = time.time()
    model.train()
    train_loss = 0
    correct = 0
    count = 0

    for start_index in range(0, N_train, batch_size):
        # Task 3
        end_index = min(start_index+batch_size, N_train-1)
        features_batch, adj_batch, idx_batch, y_batch = get_batch(G_train, y_train, start_index, end_index)
        optimizer.zero_grad()
        output = model(features_batch, adj_batch, idx_batch)
        loss = loss_function(output, y_batch)
        train_loss += loss.item() * output.size(0)
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(train_loss / count),
              'acc_train: {:.2%}'.format(correct / count),
              'time: {:.4f}s'.format(time.time() - t))
print('Optimization finished!')
# ACCURACY SHALL GO TO 90% , loss = 0.3

# Evaluates the model
model.eval()
test_loss = 0
correct = 0
count = 0
for start_index in range(0, N_test, batch_size):
    # Task 3
    end_index = min(start_index+batch_size, N_test-1)
    features_batch, adj_batch, idx_batch, y_batch = get_batch(G_test, y_test, start_index, end_index)
    output = model(features_batch, adj_batch, idx_batch)
    loss = loss_function(output, y_batch)
    test_loss += loss.item() * output.size(0)
    count += output.size(0)
    preds = output.max(1)[1].type_as(y_batch)
    correct += torch.sum(preds.eq(y_batch).double())

print('loss_test: {:.4f}'.format(test_loss / count),
      'acc_test: {:.2%}'.format(correct / count),
      'time: {:.4f}s'.format(time.time() - t))
