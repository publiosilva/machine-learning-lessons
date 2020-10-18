import statistics
import numpy as np


class KNN:
    def __init__(self, k, metric='euclidean'):
        self.k = k
        self.metrics = {
            'euclidean': lambda r, s: np.sqrt(np.sum(np.power(r - s, 2))),
            'manhattan': lambda r, s: np.sum(np.abs(r - s))
        }
        self.metric = self.metrics[metric]
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, x):
        distances = np.array([[self.metric(x_i, X_i) for X_i in self.X] for x_i in x])
        neighbor_indexes = np.array(np.argpartition(distances, self.k - 1)[:, :self.k])
        neighbor_labels = np.array([self.y[ni] for ni in neighbor_indexes])
        y = np.array([ statistics.mode(nl) for nl in neighbor_labels ])
        
        return y