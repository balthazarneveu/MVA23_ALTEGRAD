"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np
from typing import Tuple


def create_train_dataset(n_train: int = 100000, max_train_card: int = 10):  # -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros((n_train))
    for i in range(n_train):
        card = np.random.randint(1, max_train_card+1)
        X_train[i, -card:] = np.random.randint(1, max_train_card+1, size=card)
        y_train = np.sum(X_train[i, :])
    return X_train, y_train


def create_train_dataset_unicity() -> Tuple[np.ndarray, np.ndarray]:
    """Not multiset implementation

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    n_train = 100000
    max_train_card = 10
    X_train = np.empty((n_train, 10))
    y_train = np.empty(n_train)
    # Task 1
    for idx in range(n_train):
        sample = np.random.randint(1, 11, max_train_card)
        sample_set = set(sample)
        sample_set = np.array(list(sample_set))
        padded_set = np.zeros(10)
        padded_set[-len(sample_set):] = sample_set
        assert padded_set.shape[0] == 10
        X_train[idx, :] = padded_set
        y_train[idx] = padded_set.sum()
    return X_train, y_train


def create_test_dataset(n_test=200000) -> Tuple[np.ndarray, np.ndarray]:
    # Task 2
    min_test_card = 5
    max_test_card = 101
    step_test_card = 5
    cards = range(min_test_card, max_test_card, step_test_card)
    n_samples_per_card = n_test // len(cards)

    X_test = []
    y_test = []
    for card in cards:
        x = np.random.randint(1, 11, size=(n_samples_per_card, card))
        y = x.sum(axis=1)
        X_test.append(x)
        y_test.append(y)
    return X_test, y_test


if __name__ == '__main__':
    x_train, y_train = create_train_dataset()
    assert x_train.shape == (100000, 10)
    X_test, y_test = create_test_dataset()
    for idx in range(len(X_test)):
        print(X_test[idx].shape)
    # print(X_test[1].shape)
