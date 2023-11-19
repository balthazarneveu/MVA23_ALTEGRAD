"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
PART 2: Node classification
"""
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
DATA_ROOT = Path(__file__).parent/".."/"data"


def load_labelled_graph(
        graph_pth: Path= DATA_ROOT/'karate.edgelist',
        label_pth: Path = DATA_ROOT/'karate_labels.txt'
    ) -> Tuple[nx.Graph, np.ndarray]:
    """Load labelled graph

    Args:
        graph_pth (Path, optional): Path to edgelist.
        label_pth (Path, optional): Path to labels.

    Returns:
        Tuple[nx.Graph, np.ndarray]: Graph, labels
    """
    # Loads the karate network
    G = nx.read_weighted_edgelist(graph_pth, delimiter=' ', nodetype=int, create_using=nx.Graph())
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Loads the class labels
    class_labels = np.loadtxt(label_pth, delimiter=',', dtype=np.int32)
    idx_to_class_label = dict()
    for i in range(class_labels.shape[0]):
        idx_to_class_label[class_labels[i,0]] = class_labels[i,1]
    
    # Class labels as an array
    y = list()
    for node in G.nodes():
        y.append(idx_to_class_label[node])

    y = np.array(y)
    return G, y


############## Task 5
# Visualizes the karate network
def visualize_network(G: nx.Graph, labels:np.ndarray, title:str="") -> None:
    """Plot network graph

    Args:
        G (nx.Graph): graph
        labels (np.ndarray): arrays of labels (int)
        title (str, optional): Adds a title to know the label kind
    """
    nx.draw_networkx(G, node_color=labels)
    plt.title(f"Labelled graph {title}")
    plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
def deepwalk_embeddings(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    n_dim = 128
    n_walks = 10
    walk_length = 20
    model = deepwalk(G, n_walks, walk_length, n_dim)

    embeddings = np.zeros((n, n_dim))
    nodes = model.wv.index_to_key[:n]
    print(nodes)
    print(G.nodes())
    for i, node in enumerate(G.nodes()):
        embeddings[i,:] = model.wv[node]
    return embeddings


def shuffle_split_dataset(
        embeddings: np.ndarray,
        labels:np.ndarray,
        number_of_nodes,
        train_ratio = 0.8,
        seed=42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.RandomState(seed=seed).permutation(number_of_nodes)
    idx_train = idx[:int(train_ratio*number_of_nodes)]
    idx_test = idx[int(train_ratio*number_of_nodes):]

    X_train = embeddings[idx_train,:]
    X_test = embeddings[idx_test,:]

    y_train = labels[idx_train]
    y_test = labels[idx_test]
    return X_train, y_train, X_test, y_test

############## Task 7
# Trains a logistic regression classifier and use it to make predictions
def classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{accuracy=:.3f}")
    return accuracy

############## Task 8
# Generates spectral embeddings
def compute_rw_laplacian(graph: nx.Graph) -> np.ndarray:
    adj = nx.adjacency_matrix(graph)
    deg = np.sum(adj, axis=1) 
    # This matches the definition of the degree
    d = 1./deg
    laplacian = eye(adj.shape[0])- diags(d).dot(adj) # keeps the spasity!
    return laplacian

def spectral_embeddings(graph: nx.Graph, k=2) -> np.ndarray:
    laplacian = compute_rw_laplacian(graph)
    sorted_eigen_values, eigen_vectors = eigs(
        laplacian,
        which="SR", # SR for smallest real part
        k=k
    )
    sorted_eigen_values = sorted_eigen_values.real
    u_matrix = eigen_vectors.real # matrix of k sorted eigen vector
    print(u_matrix.shape)
    return u_matrix


def study_classifiers(G: nx.Graph, embedding_list:Dict[str, np.ndarray], labels: np.ndarray, n_runs=10):
    train_ratios = np.linspace(0.2, 0.9, 30)
    plt.figure(figsize=(5, 5))
    for embed_type_index, (name, embedding) in enumerate(embedding_list.items()):
        color = "gcr"[embed_type_index]
        accuracies = np.zeros_like(train_ratios)
        std_dev = np.zeros_like(train_ratios)
        mini, maxi = np.zeros_like(train_ratios), np.zeros_like(train_ratios)
        for idx, train_ratio in enumerate(train_ratios):
            current_accuracies = np.zeros(n_runs)
            for idx_seed, seed in enumerate(range(42, 42+n_runs)):
                X_train, y_train, X_test, y_test = shuffle_split_dataset(
                    embedding, labels,
                    G.number_of_nodes(),
                    train_ratio=train_ratio,
                    seed=seed
                )
                current_accuracies[idx_seed] = classification(X_train, y_train, X_test, y_test)
            accuracies[idx] = np.mean(current_accuracies)
            std_dev[idx] = np.std(current_accuracies)
            mini[idx], maxi[idx] = np.min(current_accuracies), np.max(current_accuracies)
            plt.plot([train_ratio, train_ratio], [mini[idx], maxi[idx]], "-^", alpha=0.2, color=color)
        plt.plot(train_ratios, accuracies, "-o", color=color, label=name, linewidth=5)
        
        # plt.errorbar(train_ratios, accuracies, std_dev, linestyle='None', marker='^')
    plt.xlim(0, 1.)
    plt.ylim(0, 1.01)
    plt.xlabel("Training ratio")
    plt.ylabel("Classifier accuracy")
    plt.grid()
    plt.legend()
    plt.show()

def main(viz=True):
    G, y_true = load_labelled_graph()
    emb_lap = spectral_embeddings(G)
    if viz:
        visualize_network(G, y_true, title="Groundtruth")
    emb_dw = deepwalk_embeddings(G)
    study_classifiers(
        G,
        {
            "Deepwalk embeddings": emb_dw,
            "Laplacian embeddings": emb_lap,
        },
        y_true
    )
    

if __name__ == "__main__":
    main(viz=False)