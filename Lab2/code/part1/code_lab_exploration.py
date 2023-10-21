"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pprint
from typing import Callable
import functools


def task(func: Callable):
    """Wrapper to split the results between tasks while printing"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(40 * "_" + f" {func.__name__}" + 40 * "_")
        return func(*args, **kwargs)
    return wrapper


############## Task 1
def load_graph(edge_path: Path) -> nx.Graph:
    """
    Load the network data into an undirected graph G
    using the read edgelist() function of NetworkX.
    """
    graph = nx.read_edgelist(edges_file)
    return graph

def get_stats(graph: nx.Graph):
    stats = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges)
    }
    return stats

@task
def task_1(graph: nx.Graph) -> dict:
    """
    Furthermore, compute and print
    the following network characteristics: (1) number of nodes, (2) number of edges
    Return stats
    """
    stats = get_stats(graph)
    n = stats["total_nodes"]
    assert stats["total_edges"] <= n*(n-1)//2, "max number of edges not respected" # Question 1.1
    pprint.pprint(stats, width=10)
    print(f"max number of edges {n*(n-1)//2}")
    return stats


############## Task 2
@task
def task_2(graph: nx.Graph, stats=None):
    """Print the number of connected components. 
    If the graph is not connected, 
    Retrieve the largest connected component subgraph (also known as giant connected component)
    Find the number of nodes and edges of the largest connected component 
    and examine to what fraction of the whole graph they correspond
    """
    if stats is None:
        stats = get_stats(graph)
    connected_components_list = sorted(nx.connected_components(graph), key=len, reverse=True)
    # sort on the length of components, first element is the largest
    number_of_connected_components = len(connected_components_list)
    stats["total_connected_components"] = number_of_connected_components
    if number_of_connected_components>1:
        print(f"Graph has {number_of_connected_components} connected components")
    giant_connected_component = graph.subgraph(connected_components_list[0])
    stats_giant_component = get_stats(giant_connected_component)
    print("Largest connected component:")
    pprint.pprint(stats_giant_component, width=10)
    for key in stats_giant_component.keys():
        print(f"{stats_giant_component[key]} {key}" +
              f" represent {stats_giant_component[key]/stats[key]*100:.2f}%" +
              " of the graph")

############## Task 3
@task
def task_3(graph: nx.Graph):
    degree_sequence = [graph.degree(node) for node in graph.nodes()]
    pass


############## Task 4
@task
def task_4(graph: nx.Graph):
    pass

############## Task 5
@task
def task_5(graph: nx.Graph):
    pass

if __name__ == "__main__":
    # DATASET_FOLDER = Path("code/datasets")
    dataset_folder = Path(__file__).parent/".."/"datasets"
    edges_file = dataset_folder/"CA-HepTh.txt"
    graph = load_graph(edges_file)
    stats = {}
    stats = task_1(graph)
    task_2(graph, stats=stats)
    task_3(graph)
    task_4(graph)
    task_5(graph)