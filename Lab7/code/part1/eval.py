"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch
from train import DEEPSET, LSTMSET, RESULTS_ROOT
from train import get_results_path
import logging
from utils import create_test_dataset
from models import DeepSets, LSTM
import numpy as np


def get_checkpoint(saved_models_path):
    logging.info("Loading DeepSets checkpoint!")
    checkpoint_deepset = torch.load(saved_models_path[DEEPSET])
    logging.info("Loading LSTM checkpoint!")
    checkpoint_lstm = torch.load(saved_models_path[LSTMSET])
    return checkpoint_deepset["state_dict"], checkpoint_lstm["state_dict"]


def generate_test_set():
    # Generates test data
    X_test, y_test, cards = create_test_dataset(step_test_card=5)
    cards = [X_test[i].shape[1] for i in range(len(X_test))]
    n_samples_per_card = X_test[0].shape[0]
    n_digits = 11
    return X_test, y_test, cards, n_samples_per_card, n_digits


def evaluate(results_path=RESULTS_ROOT, **kwargs):
    saved_models_path = get_results_path(results_path)
    # saved_models_path = get_results_path(results_path, name=f"{epoch}")
    # if not saved_models_path[DEEPSET].exists() or not saved_models_path[LSTMSET].exists():
    #     logging.warning("No model found, training models...")
    #     training_loop(results_path, **kwargs)
    # Initializes device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparameters
    batch_size = 64
    embedding_dim = 128
    hidden_dim = 64

    X_test, y_test, cards, n_samples_per_card, n_digits = generate_test_set()
    checkpoint_deepset, checkpoint_lstm = get_checkpoint(saved_models_path)
    # Retrieves DeepSets model
    deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)

    deepsets.load_state_dict(checkpoint_deepset)
    deepsets.eval()

    # Retrieves LSTM model
    lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
    lstm.load_state_dict(checkpoint_lstm)
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
            y_pred_deepsets = y_pred_deepsets
            y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

            acc_deepsets = accuracy_score(y_test[i], np.round(y_pred_deepsets))
            mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)

            results[DEEPSET]['acc'].append(acc_deepsets)
            results[DEEPSET]['mae'].append(mae_deepsets)

            # Task 6
            y_pred_lstm = torch.cat(y_pred_lstm)
            y_pred_lstm = torch.round(y_pred_lstm)
            y_pred_lstm = y_pred_lstm.detach().cpu().numpy()

            acc_lstm = accuracy_score(y_test[i], np.round(y_pred_lstm))
            mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
            results[LSTMSET]['acc'].append(acc_lstm)
            results[LSTMSET]['mae'].append(mae_lstm)

    return results


def plot_results(results_dict, extra_tile=""):
    # Task 7
    plt.figure(figsize=(10, 5))
    plt.grid()
    for metric, metric_pretty_name in [("acc", "accuracy"), ("mae", "MAE")]:
        for model_name, results in results_dict.items():
            label = f'{model_name}'
            # MAE={results[DEEPSET]["mae"]
            # {results[LSTMSET]["mae"]}
            plt.plot(results["cardinals"], results[DEEPSET][metric], "-o",
                     alpha=results["epoch"]/19.,
                     #  color="green",
                     label=None if label is None else (label + f' DeepSets'))
            plt.plot(results["cardinals"], results[LSTMSET][metric], "--o",
                     alpha=results["epoch"]/19.,
                     color="cyan",
                     label=None if label is None else (label + f' LSTM '))
        plt.title(f"Evolution of the {metric_pretty_name} with the cardinality of the sets"+extra_tile)
        plt.grid()
        plt.legend(loc='lower right')
        plt.xlabel("Maximum cardinality of the sets")
        plt.ylabel(metric_pretty_name)
        plt.show()


if __name__ == "__main__":
    all_results = {}
    selection = range(20)
    # selection = [19]
    for n_epoch in selection:
        results = evaluate(RESULTS_ROOT/"seed=42"/f"{n_epoch}")
        results["epoch"] = n_epoch
        all_results[f"{n_epoch+1} epochs"] = results
    plot_results(all_results)
