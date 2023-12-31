"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023

"""

import numpy as np
from typing import Tuple, List, Optional

# Create a training and test dataset of integer sets for the purpose of learning to sum up integers


def create_train_dataset(
    n_train: int = 100000,
    max_train_card: int = 10,
    multiset: Optional[bool] = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a training dataset of integer sets
    Args:
        n_train (int, optional): The number of training samples to generate. Defaults to 100000.
        max_train_card (int, optional): The maximum number of cards in each training sample. Defaults to 10.
        multiset (Optional[bool], optional): Whether to generate multiset samples or forcing sample unicity.
        Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X_train, y_train


    Task 1
    ======
    ```text
    We will generate 100k training samples
    by randomly sampling between 1 and 10 digits (1 ≤ M ≤ 10) from {1, 2, . . . , 10}.

    Each set contains between 1 and 10 digits.
    where each digit is drawn from {1, 2, . . . , 10}.

    To train the models, it is necessary that all training samples have identical cardinalities.
    Therefore, we pad sets with cardinalities smaller than 10 with zeros.
    For instance, the set {4, 5, 1, 7} is represented as {0, 0, 0, 0, 0, 0, 4, 5, 1, 7}
    ```
    """
    # Task 1
    X_train = np.zeros((n_train, max_train_card))

    for idx in range(n_train):
        card = np.random.randint(1, max_train_card+1)
        if multiset:  # Multiset code
            X_train[idx, -card:] = np.random.randint(1, 10+1, size=card)
        else:  # Ensure sample unicity
            sample = np.random.randint(1, 11, max_train_card)
            sample_set = set(sample)
            sample_set = np.array(list(sample_set))
            card = min(card, len(sample_set))
            X_train[idx, -card:] = sample_set[:card]
    y_train = X_train.sum(axis=1)  # :-) use numpy vectorization for such operations
    return X_train.astype(np.int16), y_train.astype(np.int16)


def create_test_dataset(n_test=200000, step_test_card: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Create a test dataset of varying but balanced cardinalities
    The puprose is to test if model can generalize to unseen cardinalities,
    independently of the cardinality of the training set.

    Args:
        n_test (int, optional): number of test set samples. Defaults to 200000.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]], List[int]: X_test, y_test
        X_test: list of 20 * 10000 test samples of varying cardinalities
        [
            (10000, 5)
            (10000, 10)
            (10000, 15)
            ...
            (10000, 100)
        ]
        y_test = [(10000), (10000), (10000), ..., (10000)]
    TASK 2
    ======

    ```text
    With regards to the test set,
    we will generate 200,000 test samples of cardinalities
    from 5 to 100 containing again digits from {1, 2, . . . , 10}.

    Specifically we will create:
    - 10k samples with cardinalities exactly 5,
    - 10k samples with cardinalities exactly 10, and so on.
    ```
    """
    # Task 2
    min_test_card = 5
    max_test_card = 101
    
    cards = range(min_test_card, max_test_card, step_test_card)
    # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    n_samples_per_card = n_test // len(cards)  # 10000

    X_test = []
    y_test = []
    for card in cards:
        x = np.random.randint(1, 11, size=(n_samples_per_card, card))
        y = x.sum(axis=1)
        X_test.append(x)
        y_test.append(y)
    return X_test, y_test, list(cards)


if __name__ == '__main__':
    x_train, y_train = create_train_dataset(multiset=True)
    assert x_train.shape == (100000, 10)
    X_test, y_test, cards = create_test_dataset()
    for idx in range(len(X_test)):
        print(X_test[idx].shape)
    # print(X_test[1].shape)
