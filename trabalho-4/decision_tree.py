import numpy as np


def gini(y):
    label_unique, label_counts = np.unique(y, return_counts=True)
    label_count = dict(zip(label_unique, label_counts))

    return 1 - np.sum([ np.square(label_count[label_count_i] / np.shape(y)[0]) \
        for label_count_i in label_count ])


def apply_partition(X, y, feature, cutoff):
    left_indexes = np.where(np.array(X)[:, feature] <= cutoff)[0]
    right_indexes = np.where(np.array(X)[:, feature] > cutoff)[0]

    return X[left_indexes], \
            y[left_indexes], \
            X[right_indexes], \
            y[right_indexes]


def generate_tree(X, y):
    g_parent = gini(y)
    tree_node = {
        'label': None,
        'feature': None,
        'cutoff': None,
        'left': None,
        'right': None
    }
    

    if g_parent == 0:
        tree_node['label'] = np.unique(y)[0]
    else:
        X_T = np.transpose(X)
        greater_purity = 0
        X_left, y_left = None, None
        X_right, y_right = None, None
        
        for j, X_j in enumerate(X_T):
            X_j_unique = np.unique(X_j)
            feature = j

            for X_j_i in X_j_unique:
                cutoff = X_j_i
                p_X_left, p_y_left, p_X_right, p_y_right = apply_partition(X, y, feature, cutoff)
                g_child_1 = gini(p_y_left)
                g_child_2 = gini(p_y_right)
                purity = g_parent - ((np.shape(p_y_left)[0] / np.shape(y)[0]) * g_child_1 \
                                     + (np.shape(p_y_right)[0] / np.shape(y)[0]) * g_child_2)

                if purity > greater_purity:
                    tree_node['feature'] = feature
                    tree_node['cutoff'] = cutoff
                    greater_purity = purity
                    X_left, y_left = p_X_left, p_y_left
                    X_right, y_right = p_X_right, p_y_right

        tree_node['left'] = generate_tree(X_left, y_left)
        tree_node['right'] = generate_tree(X_right, y_right)

    return tree_node


def get_sample_label(decision_tree, x):    
    label = decision_tree['label']
    
    if label:
        return label
    
    feature = decision_tree['feature']
    cutoff = decision_tree['cutoff']
    
    if x[feature] <= cutoff:
        left = decision_tree['left']
        
        return get_sample_label(left, x)
    
    right = decision_tree['right']
    
    return get_sample_label(right, x)


class DecisionTree:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.decision_tree = generate_tree(X, y)
    
    def predict(self, x):
        y = []
        
        for x_i in x:
            y.append(get_sample_label(self.decision_tree, x_i))
            
        return np.array(y)
        
