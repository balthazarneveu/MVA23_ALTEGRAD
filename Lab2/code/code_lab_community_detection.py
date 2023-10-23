"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
from helper import create_graph_comparison, task, load_graph
from pathlib import Path




############## Task 6
def compute_laplacian(graph: nx.Graph) -> np.ndarray:
    node_list = graph.nodes()
    # Prefer to freeze the nodes by sanity
    # Even though this node order is taken default

    adj = nx.adjacency_matrix(graph, nodelist=node_list)
    
    deg = np.sum(adj, axis=1) # This does not match the degree distribution ... but more relevant to use!
    # deg = np.array([degree for _node, degree in nx.degree(graph, node_list)]) # This matches the definition of the degree
    d = 1./deg
    laplacian = np.eye(adj.shape[0])-np.dot(np.diag(d), adj.toarray())
    return laplacian


# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(graph:  nx.Graph, k:int) -> dict:
    laplacian = compute_laplacian(graph)  #mxm
    # print(laplacian)
    assert np.isclose(laplacian.sum(axis=1), 0.).all(), "Laplacian sum over columns shall be 0 if everything is correctly normalized" # Assert gets broken if you use the official degree definition
    eigen_values, eigen_vectors = np.linalg.eigh(laplacian) # not sorted
    # Reorder eigenvalues and eigen vectors by ascending orders.
    sorted_indices = eigen_values.argsort()
    sorted_eigen_values = eigen_values[sorted_indices]
    sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
    d = 20
    # y1 = (first row of U).T
    # 
    u_mat = sorted_eigen_vectors[:, :d] # m, d 
    print(u_mat.shape)
    # print(eigen_values, sorted_eigen_values)
    # print(eigen_vectors)
    # print(sorted_eigen_vectors)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(u_mat) #(n_samples, n_features)
    clustering = dict()
    return clustering

def task_6(graph: nx.Graph):
    spectral_clustering(graph, 5)
    




if __name__ == '__main__':
    dataset_folder = Path(__file__).parent/"datasets"
    figures_folder = Path(__file__).parent/".."/"report"/"figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    edges_file = dataset_folder/"CA-HepTh.txt"
    graph = load_graph(edges_file)     
    # task_6(graph)
    task_6(nx.star_graph(6))
############## Task 7

##################
# your code here #
##################







############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    

    
    
    return modularity



############## Task 9

##################
# your code here #
##################







