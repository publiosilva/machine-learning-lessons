import numpy as np


def RSS(y_true, y_predict):
    return np.sum(np.power((y_true - y_predict), 2))


def TSS(y_true):
    return np.sum(np.power(y_true - np.mean(y_true), 2))


def RSE(y_true, y_predict):
    return np.sqrt((1 / (np.shape(y_true)[0] - 2)) * RSS(y_true, y_predict))


def R2(y_true, y_predict):
    return 1 - RSS(y_true, y_predict) / TSS(y_true)


def MAE(y_true, y_predict):
    return np.sum(np.absolute(y_true - y_predict)) / np.shape(y_true)[0]


def MSE(y_true, y_predict):
    return (1 / np.shape(y_true)[0]) * np.sum(np.power(y_true - y_predict, 2))
