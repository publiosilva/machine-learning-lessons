import numpy as np

from utils import group_by, cov_matrix


class GaussianQuadraticDiscriminant:
    def __init__(self):
        self.p_c = {}
        self.mean_c = {}
        self.cov_matrix_c = {}
    
    def fit(self, X, y):
        self.c = np.unique(y)
        self.X_c = group_by(X, y, self.c)
        
        for key in self.X_c:
            X_c_i = np.array(self.X_c[key])
            
            self.p_c[key] = np.shape(X_c_i)[0] / np.shape(X)[0]
            self.mean_c[key] = np.mean(X_c_i, axis=0)
            self.cov_matrix_c[key] = cov_matrix(X_c_i)
    
    def predict(self, x):
        y = []
        
        for x_i in x:
            p_c_x = {}
            
            for key in self.X_c:
                mean_c = self.mean_c[key]
                cov_m_c = self.cov_matrix_c[key]

                p_c = self.p_c[key]
                p_x_c = (1 / (np.power(np.linalg.det(cov_m_c), 1/2) * np.power(2 * np.pi, np.shape(x)[1] / 2))) \
                    * np.exp((-1/2) * (x_i - mean_c).T @ np.linalg.inv(cov_m_c) @ (x_i - mean_c))

                p_c_x[key] = p_c * p_x_c
                
            y.append(float(max(p_c_x, key=lambda key: p_c_x[key])))
            
        return np.array(y)
