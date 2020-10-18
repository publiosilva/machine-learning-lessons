import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroid = []
        self.labels = []
        self.distances_to_centroid = []
        self.a = 0

    def _get_random_point(self, X):
        rp = [np.random.uniform(low=l, high=h) for l, h in [(np.min(X_i), np.max(X_i)) for X_i in X.T]]
        
        return rp
        
    def _generate_random_centroid(self, X):
        for _ in range(self.k):
            self.centroid.append(self._get_random_point(X))
        
    def _get_cluster_by_label(self, X, labels, cluster_label):
        cluster = []
        
        for i, X_i in enumerate(X):
            if labels[i] == cluster_label:
                cluster.append(X_i)
                
        return cluster
            
    def _calc_centroid(self, cluster):
        return np.mean(cluster, axis=0)
    
    def _generate_clusters(self, X):
        self.labels = []
        self.distances_to_centroid = []
        
        for X_i in X:
            d = []
            
            for c_i in self.centroid:
                d_i = np.sqrt(np.sum(np.power(X_i - c_i, 2)))
                d.append(d_i)

            self.labels.append(np.argmin(d))
            self.distances_to_centroid.append(np.min(d))
            
        new_centroid = []
        
        for i in range(self.k):
            cluster = self._get_cluster_by_label(X, self.labels, i)
            
            if np.shape(cluster)[0] == 0:
                new_centroid.append(self._get_random_point(X))
            else:
                new_centroid.append(self._calc_centroid(cluster))
            
        return new_centroid
    
    def fit(self, X):
        self._generate_random_centroid(X)
        new_centroid = self._generate_clusters(X)
        
        while not np.array_equal(new_centroid, self.centroid):
            self.centroid = new_centroid
            new_centroid = self._generate_clusters(X)
        