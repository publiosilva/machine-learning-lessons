import numpy as np


def train_test_split(X, y, test_size):
    n_train = int(np.shape(y)[0] * (1 - test_size))

    return (X[:n_train], X[n_train:], y[:n_train], y[n_train:])


def count_cross_occurrences(
    arr_a,
    arr_b,
    val_x,
    val_y,
    ):

    count = 0

    for i in range(len(arr_b)):
        if arr_a[i] == val_x and arr_b[i] == val_y:
            count += 1

    return count


def confusion_matrix(y_true, y_pred, labels):
    return np.array([[count_cross_occurrences(y_true, y_pred, i, j)
                    for j in labels] for i in labels])


class MinMaxScaler:
    def fit(self, data):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)


class MeanScaler:
    def fit(self, data):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        self.mean = data.mean(axis=0)

    def transform(self, data):
        return (data - self.mean) / (self.max - self.min)
    