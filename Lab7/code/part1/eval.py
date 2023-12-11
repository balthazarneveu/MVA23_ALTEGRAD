"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch
from train import DEEPSET, LSTMSET, RESULTS_ROOT, training_loop
from train import get_results_path
import logging
from utils import create_test_dataset
from models import DeepSets, LSTM


def evaluate(results_path=RESULTS_ROOT, **kwargs):
    saved_models_path = get_results_path(results_path)
    if not saved_models_path[DEEPSET].exists() or not saved_models_path[LSTMSET].exists():
        logging.warning("No model found, training models...")
        training_loop(results_path, **kwargs)
    # Initializes device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparameters
    batch_size = 64
    embedding_dim = 128
    hidden_dim = 64

    # Generates test data
    X_test, y_test, cards = create_test_dataset(step_test_card=5)
    cards = [X_test[i].shape[1] for i in range(len(X_test))]
    n_samples_per_card = X_test[0].shape[0]
    n_digits = 11

    # Retrieves DeepSets model
    deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
    logging.info("Loading DeepSets checkpoint!")
    checkpoint = torch.load(saved_models_path[DEEPSET])
    deepsets.load_state_dict(checkpoint['state_dict'])
    deepsets.eval()

    # Retrieves LSTM model
    lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
    logging.info("Loading LSTM checkpoint!")
    checkpoint = torch.load(saved_models_path[LSTMSET])
    lstm.load_state_dict(checkpoint['state_dict'])
    lstm.eval()

    # Load 2 models at once, evaluate them on the same sets

    # Dict to store the results
    results = {DEEPSET: {'acc': [], 'mae': []}, LSTMSET: {'acc': [], 'mae': []}}
    results["cardinals"] = cards
    with torch.no_grad():
        for i in range(len(cards)):
            y_pred_deepsets = list()
            y_pred_lstm = list()
            for j in range(0, n_samples_per_card, batch_size):
                # Task 6
                x_batch = torch.Tensor(X_test[i][j:min(j+batch_size, n_samples_per_card), :]
                                       ).to(device, dtype=torch.int64)
                y_pred_deepsets.append(deepsets(x_batch))
                y_pred_lstm.append(lstm(x_batch))
            y_pred_deepsets = torch.cat(y_pred_deepsets)
            y_pred_deepsets = torch.round(y_pred_deepsets)
            y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

            acc_deepsets = accuracy_score(y_test[i], y_pred_deepsets)
            mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)

            results[DEEPSET]['acc'].append(acc_deepsets)
            results[DEEPSET]['mae'].append(mae_deepsets)

            # Task 6
            y_pred_lstm = torch.cat(y_pred_lstm)
            y_pred_lstm = torch.round(y_pred_lstm)
            y_pred_lstm = y_pred_lstm.detach().cpu().numpy()

            acc_lstm = accuracy_score(y_test[i], y_pred_lstm)
            mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
            results[LSTMSET]['acc'].append(acc_lstm)
            results[LSTMSET]['mae'].append(mae_lstm)

    return results


def plot_results(results_dict):
    # Task 7
    plt.figure(figsize=(10, 5))
    plt.grid()
    for model_name, results in results_dict.items():
        plt.plot(results["cardinals"], results[DEEPSET]['acc'], "-o", alpha=results["epoch"]/8., label=f'{model_name} DeepSets')
        plt.plot(results["cardinals"], results[LSTMSET]['acc'], "--o", alpha=results["epoch"]/8., label=f'{model_name} LSTM')
    plt.legend()
    plt.xlabel("Maximum cardinality of the sets")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    all_results = {}
    for n_epoch in range(3, 8):
        results = evaluate(RESULTS_ROOT/f"n={n_epoch}", epochs=n_epoch)
        results["epoch"] = n_epoch
        all_results[f"{n_epoch} epochs"] = results
    plot_results(all_results)
