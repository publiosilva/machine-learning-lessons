import numpy as np

from utils import cov_matrix

class PCA:
    def __init__(self, k_components):
        self.k_components = k_components
    
    def standardize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    def fit(self, X):
        standardized_X = self.standardize(X)
        X_cov_matrix = cov_matrix(standardized_X)
        e_values, e_vectors = np.linalg.eig(X_cov_matrix)
        sorted_e_values, sorted_e_vectors = (list(t) for t in zip(*sorted(zip(e_values, e_vectors), reverse=True)))
        sorted_e_values_k, sorted_e_vectors_k = sorted_e_values[:self.k_components], sorted_e_vectors[:self.k_components]
        
        self.e_values = e_values
        self.e_values_ratio = e_values / np.sum(e_values)
        self.W = np.transpose(sorted_e_vectors_k)
        
    def fit_transform(self, X):
        return X @ self.W