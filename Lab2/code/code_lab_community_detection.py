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
import helper
from code_lab_exploration import extract_giant_component



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
            which="SR", # SR for smallest real part
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



@task
def task_6(figure_folder=None):
    """Toy example on 3 disjoint subgraphs (complete, cycle, star). k=3 clusters found"""
    n1 = 3
    n2 = 4
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
        figure_folder=figure_folder,
        fig_name="spectral_clustering_toy_example.png",
        legend="Spectral graph clustering on a toy example",
        graph_names=["Spectral graph clustering on a toy example"],
        seed=42
    )

    # Sanity check on a star graph
    # - 0 = multiplicity 1
    # - 1 = multiplicity n-2 
    # - 2 = multiplicity 1
    # S4, n=4
    # spectral_clustering(nx.star_graph(3), 2, sparse=False, debug_prints=True)
    # spectral_clustering(nx.star_graph(3), 2, sparse=False, debug_prints=True)




############## Task 7
def task_7(graph: nx.Graph, k:int = 50) -> dict:
    """Spectral clustering of the giant component of the CA-HepTh graph
    Apply the Spectral Clustering algorithm
    to the giant connected component of the CA-HepTh
    dataset, trying to identify k=50 clusters.
    """
    clusters = spectral_clustering(graph, k, sparse=True)
    from collections import Counter
    cluster_counts = Counter(clusters.values())
    print(cluster_counts)
    return clusters
    

############## Task 8
def modularity(graph: nx.Graph, clustering: dict):
    """Compute modularity value from graph G based on clustering"""
    m = len(graph.edges)
    nc = len(set(clustering.values()))
    modularity = 0
    for cluster_label in range(nc):
        community_nodes = [node for node, label in clustering.items() if label==cluster_label]
        community_graph = nx.subgraph(graph, community_nodes)
        lc = len(community_graph.edges) # number of edges withing the community.
        dc = np.array([degree for _node, degree in nx.degree(graph, community_nodes)]).sum()
        modularity+= lc/m - (dc/(2*m))**2
    return modularity

############## Task 9
@task
def task_9(graph: nx.Graph, clustering:dict, k=50):
    """Modularity computation of the giant connected component of the CA-HepTh dataset."""
    modularity(graph, clustering)
    random_clustering = {}
    for node in graph.nodes:
        random_clustering[node] = randint(0, k-1)
    modularity(graph, random_clustering)


# Question 5: numerical validation
@task
def question_5(figure_folder:Path=None):
    """Modularity computation"""
    gr1 = nx.complete_graph(range(1, 5))
    gr2 = nx.complete_graph(range(5, 9))
    gr = nx.union(gr1, gr2)
    gr.add_edge(3, 5)
    gr.add_edge(4, 6)
    cluster_left = {1: 0, 2: 0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1}
    cluster_right= {1: 0, 2: 0, 3:1, 4:0, 5:1, 6:0, 7:1, 8:0}
    mod_left = modularity(gr, cluster_left)
    mod_right= modularity(gr, cluster_right)
    create_graph_comparison(
        [gr, gr],
        node_labels=[cluster_left, cluster_right, cluster_left],
        legend=f'modularity comparison (computation using the python implementation)',
        properties=[],
        graph_names=[f"left Q={mod_left:.3f}", f"right Q={mod_right:.3f}"],
        fig_name="modularity_computation.png",
        figure_folder=figure_folder
    )
if __name__ == '__main__':
    helper.latex_mode = True
    dataset_folder = Path(__file__).parent/"datasets"
    figures_folder = Path(__file__).parent/".."/"report"/"figures"
    question_5(figure_folder=figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)
    task_6(figure_folder=figures_folder)
    edges_file = dataset_folder/"CA-HepTh.txt"
    graph = load_graph(edges_file)
    giant_component = extract_giant_component(graph)
    cluster_dict = task_7(graph)


















