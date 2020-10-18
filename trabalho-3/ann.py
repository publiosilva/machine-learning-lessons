import numpy as np


class MLP:
    def __init__(self, hl_size, learning_rate=0.001, ages=200):
        self.hl_size = hl_size
        self.learning_rate = learning_rate
        self.ages = ages
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __sigmoid_drvd(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    def fit(self, X, y):
        self.w = np.random.rand(self.hl_size, np.shape(X)[1] + 1)
        self.m = np.random.rand(self.hl_size + 1)
        
        for _ in range(self.ages):
            _X = np.c_[np.ones(np.shape(X)[0]) * -1, X]
            u_i = _X @ self.w.T
            z_i = self.__sigmoid(u_i)
            _z_i = np.c_[np.ones(np.shape(z_i)[0]) * -1, z_i]
            u_k = _z_i @ self.m
            y_bar = np.round(self.__sigmoid(u_k))
            e = y - y_bar
            
            delta_k = e * self.__sigmoid_drvd(u_k)
            delta_i = self.__sigmoid_drvd(u_i) * delta_k[:, np.newaxis] * self.m[1:]
            
            for j, _X_j in enumerate(_X):
                _X_j = _X[j]
                _z_i_j = _z_i[j]

                delta_k_j = delta_k[j]
                delta_i_j = delta_i[j]
                
                self.m += self.learning_rate * delta_k_j * _z_i_j
                self.w += self.learning_rate * delta_i_j[:, np.newaxis] @ _X_j[np.newaxis]

    
    def predict(self, x):
        _x = np.c_[np.ones(np.shape(x)[0]) * -1, x]
        u_i = _x @ self.w.T
        z_i = self.__sigmoid(u_i)
        _z_i = np.c_[np.ones(np.shape(z_i)[0]) * -1, z_i]
        u_k = _z_i @ self.m
        y_bar = np.round(self.__sigmoid(u_k))
        
        return y_bar