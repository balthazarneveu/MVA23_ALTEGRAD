import networkx as nx
import numpy as np

def compute_degree_vector(graph:  nx.Graph):
    node_list = list(graph.nodes())
    adj = nx.adjacency_matrix(graph, nodelist=node_list).toarray()
    d = np.sum(adj, axis=1)
    deg = np.array([degree for node,degree in nx.degree(graph, node_list)])
    assert (d==deg).all(), f"{np.abs(d-deg).sum()}, \nsum over columns: {d}, \ndegree function {deg}, \nadjacency=\n{adj}"


def test_degree_vector_ok():
    compute_degree_vector(nx.cycle_graph(5))
    compute_degree_vector(nx.path_graph(5))
    compute_degree_vector(nx.star_graph(5))

def test_degree_vector_nok():
    gr = nx.Graph()
    gr.add_edges_from([("a", "b"), ("b", "c"), ("a", "a")])
    compute_degree_vector(gr)