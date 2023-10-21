"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pprint


############## Task 1
def load_graph(edge_path: Path) -> nx.Graph:
    graph = nx.read_edgelist(edges_file)
    return graph

def get_stats(graph: nx.Graph):
    stats = {
        "#nodes": len(graph.nodes),
        "#edges": len(graph.edges)
    }
    return stats


def task_1(path: Path):
    graph = load_graph(path)
    stats = get_stats(graph)
    # print(stats)
    pprint.pprint(stats, width=10)

if __name__ == "__main__":
    
    # DATASET_FOLDER = Path("code/datasets")
    dataset_folder = Path(__file__).parent/".."/"datasets"
    edges_file = dataset_folder/"CA-HepTh.txt"
    task_1(edges_file)
    

############## Task 2

##################
# your code here #
##################



############## Task 3
# Degree
# degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################



############## Task 4

##################
# your code here #
##################




############## Task 5

##################
# your code here #
##################
