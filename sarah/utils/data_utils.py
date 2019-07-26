import numpy as np


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8
):
    """Split dataset into training and test.

    :param X: training data.
    :param y: labels.
    :param train_size: Float, should be between 0.0 and 1.0 and
        represent the proportion of the dataset to include in the
        train split. The rest proportion will be considered as test.
    """
    indices = np.random.permutation(x.shape[0])
    num_train_samples = int(train_size * len(indices))
    indices_train = indices[: num_train_samples]
    indices_test = indices[num_train_samples:]
    return (X[indices_train, :], y[indices_train, :]), (
        X[indices_test, :], y[indices_test, :])
