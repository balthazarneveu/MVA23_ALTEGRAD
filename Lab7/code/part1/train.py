"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from utils import create_train_dataset
from models import DeepSets, LSTM
DEEPSET, LSTMSET = "deepset", "lstm"
RESULTS_ROOT = Path('__results')


def get_results_path(results_path=RESULTS_ROOT) -> dict:
    results_path.mkdir(parents=True, exist_ok=True)
    saved_models_path = {
        DEEPSET: results_path/'model_deepsets.pth.tar',
        LSTMSET: results_path/'model_lstm.pth.tar'
    }
    return saved_models_path


def training_loop(
    results_path=RESULTS_ROOT,
    epochs=20,
    batch_size=64,
    embedding_dim=128,
    hidden_dim=64,
    learning_rate=0.001
) -> dict:
    saved_models_path = get_results_path(results_path)
    # Initializes device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Hyperparameters

    # Generates training data
    X_train, y_train = create_train_dataset()
    n_train = 100000
    n_digits = 11

    # Initializes DeepSets model and optimizer
    deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(deepsets.parameters(), lr=learning_rate)
    loss_function = nn.L1Loss()

    if not saved_models_path[DEEPSET].exists():
        # Trains the DeepSets model
        for epoch in range(epochs):
            t = time.time()
            deepsets.train()

            train_loss = 0
            count = 0
            permut = np.random.permutation(n_train)
            for i in range(0, n_train, batch_size):

                # Task 5
                optimizer.zero_grad()
                permut_indexes = permut[i:min(i+batch_size, n_train-1)]
                x_batch = torch.Tensor(X_train[permut_indexes, :]).to(device, dtype=torch.int64)
                y_batch = torch.Tensor(y_train[permut_indexes]).to(device, dtype=torch.int64)
                output = deepsets(x_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * output.size(0)
                count += output.size(0)

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(train_loss / count),
                  'time: {:.4f}s'.format(time.time() - t))

        # Stores DeepSets model into disk
        train_dict_deepset = {
            'state_dict': deepsets.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(train_dict_deepset, saved_models_path[DEEPSET])

        print("Finished training for DeepSets model")
    else:
        train_dict_deepset = torch.load(saved_models_path[DEEPSET])

    # Initializes LSTM model and optimizer
    lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)
    loss_function = nn.L1Loss()
    if not saved_models_path[LSTMSET].exists():
        # Trains the LSTM model
        for epoch in range(epochs):
            t = time.time()
            lstm.train()

            train_loss = 0
            count = 0
            permut = np.random.permutation(n_train)
            for i in range(0, n_train, batch_size):

                # Task 5
                permut_indexes = permut[i:min(i+batch_size, n_train-1)]
                x_batch = torch.Tensor(X_train[permut_indexes, :]).to(device, dtype=torch.int64)
                y_batch = torch.Tensor(y_train[permut_indexes]).to(device)

                optimizer.zero_grad()
                output = lstm(x_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * output.size(0)
                count += output.size(0)

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(train_loss / count),
                  'time: {:.4f}s'.format(time.time() - t))

        # Stores LSTM model into disk
        train_dict_lstm = {
            'state_dict': lstm.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(train_dict_lstm, saved_models_path[LSTMSET])

        print("Finished training for LSTM model")
    else:
        train_dict_lstm = torch.load(saved_models_path[DEEPSET])
    return {DEEPSET: train_dict_lstm, LSTMSET: train_dict_lstm}


if __name__ == '__main__':
    training_loop()
