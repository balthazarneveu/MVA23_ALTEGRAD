"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pprint
from typing import Callable, Optional
import functools
latex_mode = False

# ______________________________________________________________________________
# REPORT HELPERS
# ______________________________________________________________________________
def task(func: Callable):
    """Wrapper to split the results between tasks while printing
    When using the latex flag, it automatically adds the right latex
    language words:
    -Section name:
        - task_xx is deduced from the function name
        - description comes from the first line of the docstring
    - all prints will be translated to vertbatim so it looks like a command line log
    
    Author: Balthazar Neveu
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global latex_mode
        if latex_mode:
            sec_name = " ".join(func.__name__.split("_"))
            sec_name = sec_name.capitalize()
            sec_name += " : " + func.__doc__.split("\n")[0]
            print(r"\subsection*{%s}"%(sec_name))
            print(r"\begin{verbatim}")
            
        else:
            # Command line style
            print(40 * "-" + f" {func.__name__} " + 40 * "-")
            print(func.__doc__.split("\n")[0])
            print((len(func.__name__) + 2+ 80) * "_")
        results = func(*args, **kwargs)
        if latex_mode:
            print(r"\end{verbatim}")
        print("\n")
        return results
    return wrapper

def include_latex_figure(fig_name, legend, close_restart_verbatim=True, label=None):
    """Latex code to include a matplotlib generated figure"""
    fig_desc = [
        r"\end{verbatim}" if close_restart_verbatim else "",
        r"\begin{figure}[ht]",
        "\t"+r"\centering",
        "\t"+r"\includegraphics[width=.6\textwidth]{figures/%s}"%fig_name,
        "\t"+r"\caption{%s}"%legend,
        ("\t"+ r"\label{fig:%s}"%label) if label is not None else "",
        r"\end{figure}",
        r"\begin{verbatim}" if close_restart_verbatim else ""
    ]
    print("\n".join(fig_desc))
# ______________________________________________________________________________


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
    
def save_graph(figure_folder=None, fig_name=None, legend="", close_restart_verbatim=True):
    if figure_folder is not None:
        assert fig_name is not None
        fig_path = figure_folder/fig_name
        plt.savefig(fig_path)
        global latex_mode
        if not latex_mode:
            print(f"Saving {fig_path}")
        
        if latex_mode:
            include_latex_figure(
                fig_name,
                legend,
                close_restart_verbatim=close_restart_verbatim,
                label=fig_name.replace(".png", "")
            )
    else:
        plt.show()

############## Task 5
@task
def task_5(graph: nx.Graph):
    """Global clustering coefficient
    """
    print(nx.transitivity(graph))



############## Question 2
def visualize_graph(
        ax: plt.Axes,
        graph: nx.Graph,
        title,
        color='lightgreen',
        properties=["degree"]):
    """Utility function to visualize a graph with a given title."""
    
    nx.draw(
        graph,
        # pos,
        ax=ax,
        with_labels=True,
        node_size=700,
        node_color=color,
        font_size=15
    )
    if "degree" in properties:
        degree_distribution = [f"{graph.degree(node):d}" for node in graph.nodes()]
        title += "\nDegree distribution:" + " ".join(degree_distribution)
    if "transitivity" in properties:
        transitivity = nx.transitivity(graph)
        title += f"\nTransitivity: {transitivity:.3f}"
    ax.set_title(title)

def create_graph_comparison(
        graph_def = [
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')],
            [('w', 'x'), ('x', 'y'), ('y', 'w'), ('z', 'z')]
        ],
        figure_folder=None,
        legend="",
        graph_names = ["Graph G1", "Graph G2"],
        colors=['lightblue', 'lightgreen'],
        fig_name="graph_comparison.png",
        properties=["degree"],
    ):
    
    fig, axs = plt.subplots(1, len(graph_def), figsize=(len(graph_def)*5, 5))
    for index, graph_x_def in enumerate(graph_def):
        graph_x = nx.Graph()
        graph_x.add_edges_from(graph_x_def)
        visualize_graph(
            axs[index],
            graph_x,
            graph_names[index],
            color=colors[index%2],
            properties=properties
        )
    save_graph(
        figure_folder=figure_folder,
        fig_name=fig_name,
        legend=legend,
        close_restart_verbatim=False,
    )

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
    # DATASET_FOLDER = Path("code/datasets")
    dataset_folder = Path(__file__).parent/".."/"datasets"
    figures_folder = Path(__file__).parent/".."/".."/"report"/"figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    edges_file = dataset_folder/"CA-HepTh.txt"

    latex_mode = True

    # graph = load_graph(edges_file)
    # stats = {}
    # stats = task_1(graph)
    # task_2(graph, stats=stats)
    # task_3(graph)
    # task_4(graph, output_path=None if not latex_mode else figures_folder)
    # task_5(graph)

    # question_2(figure_folder=figures_folder)
    question_3(figure_folder=figures_folder)