"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pprint
from typing import Optional
from helper import task, create_graph_comparison, save_graph, load_graph
import helper

############## Task 1


def get_stats(graph: nx.Graph):
    stats = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges)
    }
    return stats

@task
def task_1(graph: nx.Graph) -> dict:
    """Basic graph statistics extraction
    Extract basic statitics from the graphs
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
def extract_giant_component(graph: nx.Graph, stats={}):
    connected_components_list = sorted(nx.connected_components(graph), key=len, reverse=True)
    # sort on the length of components, first element is the largest
    number_of_connected_components = len(connected_components_list)
    stats["total_connected_components"] = number_of_connected_components
    if number_of_connected_components>1:
        print(f"Graph has {number_of_connected_components} connected components")
    giant_connected_component = graph.subgraph(connected_components_list[0])
    return giant_connected_component

@task
def task_2(graph: nx.Graph, stats=None):
    """Extract the connected components of the graph
    Print the number of connected components. 
    If the graph is not connected, 
    Retrieve the largest connected component subgraph (also known as giant connected component)
    Find the number of nodes and edges of the largest connected component 
    and examine to what fraction of the whole graph they correspond
    """
    if stats is None:
        stats = get_stats(graph)
    giant_connected_component = extract_giant_component(graph, stats=stats)
    stats_giant_component = get_stats(giant_connected_component)
    print("Largest connected component:")
    pprint.pprint(stats_giant_component, width=10)
    for key in stats_giant_component.keys():
        print(f"{stats_giant_component[key]} {key}" +
              f" represent {stats_giant_component[key]/stats[key]*100:.2f}%" +
              " of the graph")
    return giant_connected_component
############## Task 3
def stats_array(sequence:np.ndarray, stat: dict={}, title="degree_of_nodes"):
    stat[title] = {}
    for stat_type in ["min", "max", "median", "mean"]:
        stat[title][stat_type] = getattr(np, stat_type)(sequence)
    return stat

@task
def task_3(graph: nx.Graph):
    """Statistics on the degrees of the nodes of the graph
    Find and print the minimum, maximum, median and mean degree
    of the nodes of the graph
    """
    degree_sequence = [graph.degree(node) for node in graph.nodes()]
    stats_degree = stats_array(np.array(degree_sequence))
    pprint.pprint(stats_degree, width=-1)
    return stats_degree


############## Task 4
@task
def task_4(graph: nx.Graph, output_path: Optional[Path]=None):
    """Degree histogram plot
    Plot the degree histogram using the matplotlib library of Python
    (Hint: use the degree histogram() function that returns 
    a list of the frequency of each degree value).
    Produce again the plot using log-log axis
    """
    degree_count = nx.degree_histogram(graph)
    degree_density = np.array(degree_count).astype(np.float64)
    degree_density /= degree_density.sum()
    degree_axis = np.arange(len(degree_count))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for idx, graph_type in enumerate(["", "log"]):
        ax = axs[idx]
        ax.plot(degree_axis[1:], degree_density[1:])
        if idx == 1:
            ax.set_xscale(graph_type)
            ax.set_yscale(graph_type)
        ax.set_xlabel(graph_type + " Degree of node")
        ax.set_ylabel(graph_type + " Density")
        ax.set_title(f"{graph_type}-{graph_type} histogram of the degrees of the nodes")
        ax.grid()
    save_graph(
        figure_folder=output_path,
        fig_name="histogram_degree_of_nodes.png",
        legend="Histogram of degrees of the nodes",
    )


############## Task 5
@task
def task_5(graph: nx.Graph):
    """Global clustering coefficient
    """
    print(f"Graph clustering coefficient: {nx.transitivity(graph):.3f}")



############## Question 2
@task
def question_2(figure_folder=None):
    r"""2 graphs having the same degree distribution $\not \implies$ isomorphic$
    """
    create_graph_comparison(
        graph_def = [
            [('a', 'b'), ('b', 'c'), ('c', 'a')],
            [('w', 'w'), ('x', 'x'), ('y', 'y')]
        ],
        figure_folder=figure_folder,
        fig_name="graph_triangle_vs_three_single_loops.png",
        legend="Graphs G1 is a triangle and G2 is made of 3 isolated nodes with a self loop."+ 
        "They have the same degree distribution (every node has a degree of 2)"+
        "but are not isomorphic to each other.",
    )
    create_graph_comparison(
        graph_def = [
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')],
            [('w', 'x'), ('x', 'y'), ('y', 'w'), ('z', 'z')]
        ],
        figure_folder=figure_folder,
        legend="G1 is a rectangle, G2 is made of a triangle and a single node with a self loop.",
        fig_name="graph_rect_vs_triangle_plus_single_loop.png")
        

    create_graph_comparison(
        graph_def = [
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'a')],
            [('u', 'v'), ('v', 'w'), ('w', 'u'), ('x', 'y'), ('y', 'z'), ('z', 'x')],
        ],
        figure_folder=figure_folder,
        legend="G1 is an hexagon, it has 6 edges. G2 has 2 separate triangles."+
        "All nodes have a degree of 2, G1 and G2 have the same degree histograms." +
        "But they are not isomorphic to each other",
        fig_name="graph_compare_triangle_hexagon.png"
    )
    create_graph_comparison(
        graph_def = [
            [('a', 'b'), ('b', 'c'), ('c', 'a'), ('b', 'd'), ('d', 'c'), ('d', 'a')],
            [('u', 'v'), ('u', 'u'), ('v', 'v'), ('w', 'x'), ('w', 'w'), ('x', 'x')],
        ],
        figure_folder=figure_folder,
        fig_name="graph_compare_quad.png",
        legend="Counter example where all nodes have a degree of 3."+ 
        "G1=(a-b, b-c, c-a, b-d, d-c, d-a) is a rectangle with its diagonals" + 
        "G2=(u-v, u-u, v-v, w-x, w-w, x-x) are 2 segment where the end nodes have self loops"
    )
    pass

############## Question 3
@task
def question_3(figure_folder=None):
    r"""n-cycle graphs
    """
    create_graph_comparison(
        graph_def = [
            [('a', 'b'), ('b', 'c'), ('c', 'a')],
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')],
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'a')],
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('f', 'a')]
        ],
        graph_names = ["Graph C3", "Graph C4", "Graph C5", "Graph C6"],
        figure_folder=figure_folder,
        fig_name="cycle_graphs.png",
        legend="Transitivity of $C_n$ cycle graphs becomes 0 if $n\seq4$",
        properties=["transitivity"]
    )

if __name__ == "__main__":
    dataset_folder = Path(__file__).parent/"datasets"
    figures_folder = Path(__file__).parent/".."/"report"/"figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    edges_file = dataset_folder/"CA-HepTh.txt"
    # To trigger latex dumps, use helper.latex_mode = True
    # helper.latex_mode = True

    graph = load_graph(edges_file)
    stats = {}
    stats = task_1(graph)
    task_2(graph, stats=stats)
    task_3(graph)
    task_4(graph, output_path=None if not helper.latex_mode else figures_folder)
    task_5(graph)

    # question_2(figure_folder=figures_folder)
    # question_3(figure_folder=figures_folder)