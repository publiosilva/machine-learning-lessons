import numpy as np


class SingleLabelLogisticRegression:
    def __init__(self, learning_rate=0.001, ages=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.ages = ages
        self.threshold = threshold
        self.b = [0.5]
    
    def fit(self, X, y):
        for _ in range(np.shape(X)[1]):
            self.b.append(0.5)

        ones = np.ones(np.shape(X)[0])
        _X = np.c_[ones, X]
        
        for _ in range(self.ages):
            e = y - (1 / (1 + np.exp([-np.transpose(self.b) @ _Xi for _Xi in _X])))
            
            self.b += self.learning_rate * (1 / np.shape(_X)[0]) * np.sum((e * _X.T).T, axis=0)
    
    def predict(self, x):
        ones = np.ones(np.shape(x)[0])
        _x = np.c_[ones, x]
        
        y = 1 / (1 + np.exp([-np.transpose(self.b) @ _xi for _xi in _x]))
        
        return np.array([1. if _yi > self.threshold else 0. for _yi in y])
    