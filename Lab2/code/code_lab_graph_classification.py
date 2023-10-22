"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import List, Tuple
from helper import create_graph_comparison, task
from pathlib import Path
import helper

############## Task 10
# Generate simple dataset
def create_dataset() -> Tuple[List[nx.Graph], List[int], List[str]]:
    graph_list = list()
    graph_list += [nx.cycle_graph(n) for n in range(3, 103)]
    graph_list += [nx.path_graph(n) for n in range(3, 103)]
    y = 100*[0] + 100*[1]
    assert len(y) == len(graph_list)
    graph_names = [f"Cycle {idx}" for idx in range(3, 103)] + [f"Path {idx}" for idx in range(3, 103)]
    return graph_list, y, graph_names


def initialize_dataset():
    # Prepare dataset
    Gs, y, _graph_names = create_dataset()
    G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)
    return G_train, G_test, y_train, y_test


# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


############## Task 11
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    ##################
    # your code here #
    ##################


    phi_test = np.zeros((len(G_test), 4))
    
    ##################
    # your code here #
    ##################


    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test

###------------------------------------------- VISUALIZATION -----------------------------
# VISUALIZE A FEW GRAPHS
@task
def task_10(figure_folder=None):
    """Cycle and graph dataset creation"""
    Gs, y, graph_names = create_dataset()
    np.random.seed(46)
    random_index = 100*np.random.randint(0, 2, size=6) + np.random.randint(0, 20, size=6)
    create_graph_comparison(
        [Gs[idx] for idx in random_index], 
        graph_names=[graph_names[idx] for idx in random_index], properties=[],
        figure_folder=figure_folder,
        fig_name ="cycle_and_paths_dataset.png",
        legend="Dataset is made of cycles $C_n$ and paths $P_n$"
    )

### QUESTION 6
@task
def question_6(figure_folder=None):
    """Shortest path kernel"""
    # P4 graph progressively building the shortest path
    P4_graph = nx.Graph()
    P4_graph.add_weighted_edges_from([
        ("A", "B", 1),
        ("B", "C", 1),
        ("C", "D", 1)
    ]
    )
    P4_graph_sp2 = P4_graph.copy()
    P4_graph_sp2.add_weighted_edges_from(
        [("A", "C", 2), ("B","D", 2)]
    )
    P4_graph_sp3 = P4_graph_sp2.copy()
    P4_graph_sp3.add_weighted_edges_from(
        [("A", "D", 3)]
    )
    graph_shortest_paths = [
        P4_graph,
        P4_graph_sp2,
        P4_graph_sp3
    ]
    create_graph_comparison(
        graph_shortest_paths,
        graph_names=[
            "P4  shortest paths \nlength=1 (init)",
            "P4  shortest paths \nlength<=2",
            "P4  shortest paths\nlength<=3"
        ],
        properties=[],
        figure_folder=figure_folder,
        fig_name ="P4_shortest_paths_computation.png",
        legend="Path graph $P_4$, $\phi(P_4)=[3, 2, 1, 0 ...]^T$",
        weights=True
    )



    S4_graph = nx.Graph()
    S4_graph.add_weighted_edges_from([
        ("S1", "C", 1),
        ("S2", "C", 1),
        ("S3", "C", 1)
    ]
    )
    S4_graph_sp2 = S4_graph.copy()
    S4_graph_sp2.add_weighted_edges_from(
        [("S1", "S2", 2), ("S2", "S3", 2), ("S1", "S3", 2)]
    )
    graph_shortest_paths = [
        S4_graph,
        S4_graph_sp2,
    ]
    create_graph_comparison(
        graph_shortest_paths,
        graph_names=[
            "S4  shortest paths \nlength=1 (init)",
            "S4  shortest paths \nlength<=2",
        ],
        properties=[],
        figure_folder=figure_folder,
        fig_name ="S4_shortest_paths_computation.png",
        legend="Star graph $S_4$, $\phi(S_4)=[3, 3, 0, 0 ...]^T$",
        weights=True
    )


if __name__ == '__main__':
    # DATASET INITIALIZATION
    G_train, G_test, y_train, y_test = initialize_dataset()
    K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

    # VISUALIZATION
    extra_visualizations = True
    if extra_visualizations:
        helper.latex_mode = True
        figures_folder = Path(__file__).parent/".."/"report"/"figures"
        figures_folder.mkdir(parents=True, exist_ok=True)
        task_10(figure_folder=figures_folder)
        question_6(figure_folder=figures_folder)









############## Task 12

##################
# your code here #
##################




############## Task 13

##################
# your code here #
##################

