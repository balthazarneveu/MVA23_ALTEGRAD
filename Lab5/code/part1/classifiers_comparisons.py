from node_classification import *

def study_classifiers(G: nx.Graph, embedding_list:Dict[str, np.ndarray], labels: np.ndarray, n_runs=10):
    train_ratios = np.linspace(0.3, 0.9, 30)
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
    emb_dw = deepwalk_embeddings(G)
    study_classifiers(
        G,
        {
            "Deepwalk embeddings": emb_dw,
            "Laplacian embeddings": emb_lap,
        },
        y_true,
        n_runs=1000
    )
    

if __name__ == "__main__":
    main(viz=False)