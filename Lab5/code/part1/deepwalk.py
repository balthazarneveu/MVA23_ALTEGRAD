"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G: nx.Graph, source_node: int, walk_length: int):
    walk = [source_node]
    node = source_node
    for _ in range(walk_length-1):
        neighors = list(G.neighbors(node))
        degree = len(neighors)
        degree, G.degree(node)-1
        node = neighors[np.random.choice(degree)]
        walk.append(node)
    assert len(walk)==walk_length
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G: nx.Graph, num_walks: int, walk_length: int):
    walks = []
    for i in range(num_walks):
        for node in G.nodes():
            walks.append(random_walk(G, node, walk_length))
    permuted_walks = np.random.permutation(walks)
    return permuted_walks.tolist()


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
