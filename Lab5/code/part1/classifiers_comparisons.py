from node_classification import *
from tqdm import tqdm
def study_classifiers(G: nx.Graph, embedding_list:Dict[str, np.ndarray], labels: np.ndarray, n_runs=10):
    train_ratios = np.linspace(0.02, 0.95, 50)
    plt.figure(figsize=(5, 5))
    for embed_type_index, (name, embedding) in enumerate(embedding_list.items()):
        offset_viz = 0.005*embed_type_index
        color = "gcr"[embed_type_index]
        accuracies = np.ones_like(train_ratios)*np.nan
        std_dev = np.ones_like(train_ratios)*np.nan
        mini, maxi = np.ones_like(train_ratios)*np.nan, np.ones_like(train_ratios)*np.nan
        for idx, train_ratio in tqdm(enumerate(train_ratios), total=len(train_ratios)):
            current_accuracies = np.ones(n_runs)*np.nan
            for idx_seed, seed in enumerate(range(42, 42+n_runs)):
                X_train, y_train, X_test, y_test = shuffle_split_dataset(
                    embedding, labels,
                    G.number_of_nodes(),
                    train_ratio=train_ratio,
                    seed=seed
                )
                if not (1 in y_train and 0 in y_train):
                    continue
                current_accuracies[idx_seed] = classification(X_train, y_train, X_test, y_test)
            accuracies[idx] = np.mean(current_accuracies)
            std_dev[idx] = np.std(current_accuracies)
            mini[idx], maxi[idx] = np.min(current_accuracies), np.max(current_accuracies)
            plt.plot(
                [train_ratio+offset_viz, train_ratio+offset_viz],
                [mini[idx], maxi[idx]], "-^",
                alpha=0.2,
                color=color,
                label="min-max" if idx ==0 else None
            )
            plt.plot(
                [train_ratio+offset_viz, train_ratio+offset_viz], 
                [accuracies[idx]-std_dev[idx], (accuracies[idx]+std_dev[idx]).clip(None, 1)],
                "-o", alpha=0.3,
                linewidth=2,
                color=color,
                label="standard deviation" if idx ==0 else None
            )
        plt.plot(train_ratios+offset_viz, accuracies, "-o", color=color, label=name, linewidth=5)
        
        # plt.errorbar(train_ratios, accuracies, std_dev, linestyle='None', marker='^')
    plt.xlim(0, 1.)
    plt.ylim(0, 1.01)
    plt.xlabel("Labelled training data ratio")
    plt.ylabel("Classifier accuracy")
    plt.title(f"Comparison of the strength of unsupervised nodes embedding\nevaluated on a weakly supervised graph classification task - {n_runs} runs")
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
        # n_runs=10,
        n_runs=1000
    )
    

if __name__ == "__main__":
    main(viz=False)