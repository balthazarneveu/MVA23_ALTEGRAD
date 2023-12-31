"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
Part 1 : Deep walk embeddings used to "summarize" a graph visualization.
=> put DeepWalk embeddings in a low dimensional space using T-SNE
"""

import networkx as nx
import numpy as np
from deepwalk import deepwalk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import Word2Vec
# Loads the web graph
G = nx.read_weighted_edgelist(
    Path(__file__).parent/'../data/web_sample.edgelist',
    delimiter=' ',
    create_using=nx.Graph()
)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


############## Task 3
# Extracts a set of random walks from the web graph and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20

model = deepwalk(G, n_walks, walk_length, n_dim)


############## Task 4
# Visualizes the representations of the 100 nodes that appear most frequently in the generated walks
def visualize(model: Word2Vec, n: int, dim: int) -> None:
    """Use t-SNE to visualize high dimensional 
    node features a.k.a embeddings (like dim=128)
    in a nice 2D space. Keep only n points to avoid a messy graph

    Args:
        model (Word2Vec): _description_
        n (int): keep n nodes only for the visualization
        dim (int): dimension of the features (e.g. 128)
    """
    nodes = model.wv.index_to_key[:n]
    
    DeepWalk_embeddings = np.empty(shape=(n, dim))

    for idx, node in enumerate(nodes):
        DeepWalk_embeddings[idx, : ] = model.wv[node]


    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2)

    vecs_pca = my_pca.fit_transform(DeepWalk_embeddings)
    vecs_tsne = my_tsne.fit_transform(vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(vecs_tsne[:,0], vecs_tsne[:,1],s=3)
    for x, y, node in zip(vecs_tsne[:,0] , vecs_tsne[:,1], nodes):     
        ax.annotate(node, xy=(x, y), size=8)
    fig.suptitle('t-SNE visualization of node embeddings',fontsize=30)
    fig.set_size_inches(20,15)
    plt.savefig('embeddings.pdf')  
    plt.show()


visualize(model, 100, n_dim)
