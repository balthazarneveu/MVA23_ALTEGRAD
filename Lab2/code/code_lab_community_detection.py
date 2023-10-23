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
def compute_laplacian(graph: nx.Graph, sparse=False, debug_prints=False) -> np.ndarray:
    node_list = graph.nodes()
    # Prefer to freeze the nodes by sanity
    # Even though this node order is taken default

    adj = nx.adjacency_matrix(graph, nodelist=node_list)
    if debug_prints:
        print(f"ADJACENCY\n {adj.toarray()}")
    
    deg = np.sum(adj, axis=1) 
    # This sum over the columns does not match the degree distribution ... but more relevant to use!
    
    # deg = np.array([degree for _node, degree in nx.degree(graph, node_list)])
    # This matches the definition of the degree
    d = 1./deg
    if not sparse: # The classic plain matrices (slow)
        laplacian = np.eye(adj.shape[0])-np.dot(np.diag(d), adj.toarray())
    else:
        laplacian = eye(adj.shape[0])- diags(d).dot(adj) # keeps the spasity!
    return laplacian


# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(graph:  nx.Graph, k:int, d=None, sparse=True, debug_prints=False) -> dict:
    laplacian = compute_laplacian(graph, sparse=sparse, debug_prints=debug_prints)  #mxm
    if debug_prints:
        print(f"LAPLACIAN:\n{laplacian.toarray() if sparse else laplacian}")
    assert np.isclose(laplacian.sum(axis=1), 0.).all(), "Laplacian sum over columns shall be 0 if everything is correctly normalized" # Assert gets broken if you use the official degree definition
    if d is None:
        d = k
    if not sparse:
        # d = laplacian.shape[0]
        # WAY TOO SLOW WITH BIG DENSE MATRICES
        eigen_values, eigen_vectors = np.linalg.eig(laplacian) # not sorted
        # Reorder eigenvalues and eigen vectors by ascending orders.
        sorted_indices = eigen_values.argsort()
        sorted_eigen_values = eigen_values[sorted_indices]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices].real
        # Find the d most representative vectors
        u_matrix = sorted_eigen_vectors[:, :d] # m, k
    else:
        sorted_eigen_values, eigen_vectors = eigs(
            laplacian,
            which="SR", # SM for smallest real part
            k=k
        )
        sorted_eigen_values = sorted_eigen_values.real
        u_matrix = eigen_vectors.real
    if debug_prints:
        print(f"MATRIX OF EIGEN VECTORS \n{u_matrix}")
        print(f"EIGEN VALUES:\n {sorted_eigen_values}")

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(u_matrix) # (n_samples, n_features)
    predicted_labels = kmeans.predict(u_matrix)
    clustering = dict()
    node_list = list(graph.nodes())
    for idx, cluster_index in enumerate(predicted_labels):
        clustering[node_list[idx]] = cluster_index
    return clustering

def task_7(graph: nx.Graph):
    spectral_clustering(graph, 5, sparse=True)
    


def task_6():
    n1 = 4
    n2 = 6
    n_star = 6
    G1 = nx.complete_graph(n1)
    G2 = nx.cycle_graph(n2)
    G3 = nx.star_graph(n_star-1)
    # Relabel nodes of G2 to ensure they are disjoint from G1
    G2 = nx.relabel_nodes(G2, {i: i + n1 for i in range(n2)})
    # Relabel nodes of G3 to ensure they are disjoint from G1 and G2
    G3 = nx.relabel_nodes(G3, {i: i + n1+n2 for i in range(n_star)})

    # Combine the two graphs
    G = nx.union(G1, G2)
    # Combine the three graphs
    G = nx.union(G, G3)
    nodel_labels = spectral_clustering(G, 3, d=3, sparse=False, debug_prints=True)
    create_graph_comparison(
        [G],
        node_labels=[nodel_labels],
        properties=[],
        legend="Spectral graph clustering on a toy example",
        graph_names=["Spectral graph clustering on a toy example"]
    )
    
    # nodel_labels_ = spectral_clustering(G_shuffled, 2, d=2, sparse=False, debug_prints=True)
    # create_graph_comparison([G_shuffled], node_labels=[nodel_labels_])

if __name__ == '__main__':
    task_6()
    dataset_folder = Path(__file__).parent/"datasets"
    figures_folder = Path(__file__).parent/".."/"report"/"figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    edges_file = dataset_folder/"CA-HepTh.txt"
    graph = load_graph(edges_file)     
    task_7(graph)
    # task_6(nx.star_graph(20))
    # eigen_values = 
    # On a star graph
    # - 0 = multiplicity 1
    # - 1 = multiplicity n-2 
    # - 2 = multiplicity 1
    # S4, n=4
    spectral_clustering(nx.star_graph(3), 2, sparse=False, debug_prints=True)
    spectral_clustering(nx.star_graph(3), 2, sparse=False, debug_prints=True)
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







