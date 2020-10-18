import numpy as np


def cov(x1, x2):
    (x1_mean, x2_mean) = (x1.mean(), x2.mean())

    return 1 / (np.shape(x1)[0] - 1) * np.sum((x1 - x1_mean) * (x2
            - x2_mean))


def cov_matrix(data):
    data_T = np.transpose(data)

    return np.array([[cov(i, j) for i in data_T] for j in data_T])
