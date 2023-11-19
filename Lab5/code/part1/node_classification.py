"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt
from pathlib import Path
DATA_ROOT = Path(__file__).parent
# Loads the karate network
G = nx.read_weighted_edgelist(DATA_ROOT/'../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt(DATA_ROOT/'../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network
nx.draw_networkx(G, node_color=y)
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
nodes = model.wv.index_to_key[:n]
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[node]

TRAIN_RATIO = 0.8
idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(TRAIN_RATIO*n)]
idx_test = idx[int(TRAIN_RATIO*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
############## Task 8
# Generates spectral embeddings
def compute_rw_laplacian(graph: nx.Graph, sparse=False, debug_prints=False) -> np.ndarray:
    adj = nx.adjacency_matrix(graph)
    deg = np.sum(adj, axis=1) 
    # This matches the definition of the degree
    d = 1./deg
    laplacian = eye(adj.shape[0])- diags(d).dot(adj) # keeps the spasity!
    return laplacian



##################
# your code here #
##################
