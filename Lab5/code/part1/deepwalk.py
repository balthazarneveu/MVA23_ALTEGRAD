"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
Part 1 : Deep walk embeddings creations
"""
import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec
from typing import Union, List

############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(neighbors_dict: dict, source_node: Union[int, str], walk_length: int) -> List[Union[int, str]]:
    """Slightly optimized random walk 
    - using a precomputed dictionary structure to avoid using list -> remove redundant calls to list(G.neighbors(node))
    - use randint instead of np.random.choice!

    Returns:
        List[int|str]: Random walk
    """
    walk = [source_node]
    node = source_node
    for _ in range(walk_length-1):
        neighors = neighbors_dict[node]
        degree = len(neighors)
        node = neighors[randint(0, degree-1)]
        walk.append(node)
    assert len(walk)==walk_length
    return walk

############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G: nx.Graph, num_walks: int, walk_length: int) -> List[List[Union[int, str]]]:
    """Generate num_walks walks at each node of graph G with a length of walk_length.

    Args:
        G (nx.Graph): Graph
        num_walks (int): number of walks started from each node of the graph
        walk_length (int): length of the walk = number of nodes visited

    Returns:
        List[List[Union[int, str]]]: a big list with  (|V|*num_walks, walk_length) elements
        Example: Crawled Web corpus |V| 33226 , num_walks=10 -> (332260, 20)
    """
    # Precompute neighbors with a dictionary... try to speedup a bit
    neighbors_dict = {node: list(G.neighbors(node)) for node in G.nodes()}

    walks = []
    for i in range(num_walks):
        for node in G.nodes():    
            walks.append(random_walk(neighbors_dict, node, walk_length))
    permuted_walks = np.random.permutation(walks)
    return permuted_walks.tolist() # (|V|*num_walks, walk_length)


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G: nx.Graph, num_walks: int, walk_length: int, n_dim: int) -> Word2Vec:
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
