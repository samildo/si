from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


#Same imports are used above 
def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into stratified training and testing sets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.2).
    random_state : int, optional
        The seed of the random number generator (default is 42).

    Returns
    -------
    train : Dataset
        The stratified training dataset.
    test : Dataset
        The stratified testing dataset.
    """
    # Set random state
    np.random.seed(random_state)

    # Get unique class labels and their counts
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)

    # Initialize empty lists for train and test indices
    train_idxs = []
    test_idxs = []

    # Loop through unique labels
    for label, count in zip(unique_labels, label_counts):
        # Get indices for the current class
        class_idxs = np.where(dataset.y == label)[0]

        # Shuffle indices for the current class
        np.random.shuffle(class_idxs)

        # Calculate the number of test samples for the current class
        n_test = int(count * test_size)

        # Select indices for the test set
        test_class_idxs = class_idxs[:n_test]
        # Select indices for the train set
        train_class_idxs = class_idxs[n_test:]

        # Add indices to the test and train lists
        test_idxs.extend(test_class_idxs)
        train_idxs.extend(train_class_idxs)

    # Create training and testing datasets
    train = Dataset(dataset.X[train_idxs],dataset.y[train_idxs],features=dataset.features,label=dataset.label)
    test = Dataset(dataset.X[test_idxs],dataset.y[test_idxs],features=dataset.features,label=dataset.label)

    return train, test