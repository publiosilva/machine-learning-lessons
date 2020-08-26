import numpy as np


def train_test_split(X, y, test_size):
    n_train = int(np.shape(y)[0] * (1 - test_size))

    return (X[:n_train], X[n_train:], y[:n_train], y[n_train:])


def group_by(X, y, group_keys):
    groups = {}

    for key in group_keys:
        groups[str(key)] = []

    for (i, key) in enumerate(y):
        groups[str(key)].append(X[i])

    return groups


def cov(a1, a2):
    (a1_mean, a2_mean) = (a1.mean(), a2.mean())

    return 1 / (np.shape(a1)[0] - 1) * np.sum((a1 - a1_mean) * (a2
            - a2_mean))


def cov_matrix(data):
    data_T = np.transpose(data)

    return np.array([[cov(i, j) for i in data_T] for j in data_T])


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
