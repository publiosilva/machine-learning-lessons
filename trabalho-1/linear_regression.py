import numpy as np


class SimpleLinearRegressionAM():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.b_1 = np.sum((X - np.mean(X)) * (y - np.mean(y))) / \
            np.sum(np.power((X - np.mean(X)), 2))
        self.b_0 = np.mean(y) - self.b_1 * np.mean(X)

    def predict(self, x):
        return self.b_0 + self.b_1 * np.array(x)


class SimpleLinearRegressionGD():
    def __init__(self, learning_rate=0.01, ages=10):
        self.b_0 = 0
        self.b_1 = 0
        self.learning_rate = learning_rate
        self.ages = ages

    def fit(self, X, y):
        for _ in range(self.ages):
            self.b_0 += self.learning_rate * \
                (1 / np.shape(X)[0]) * np.sum(y - self.b_1 * X - self.b_0)
            self.b_1 += self.learning_rate * \
                (1 / np.shape(X)[0]) * \
                np.sum((y - self.b_1 * X - self.b_0) * X)

    def predict(self, x):
        return self.b_0 + self.b_1 * x


class MultipleLinearRegressionAM():
    def __init__(self):
        pass

    def fit(self, X, y):
        ones = np.ones(np.shape(X)[0])
        _X = np.c_[ones, X]

        self.b = np.linalg.pinv(np.transpose(_X) @ _X) @ np.transpose(_X) @ y

    def predict(self, x):
        ones = np.ones(np.shape(x)[0])
        _x = np.c_[ones, x]

        return _x @ self.b


class MultipleLinearRegressionGD():
    def __init__(self, learning_rate=0.01, ages=10):
        self.b = [0]
        self.learning_rate = learning_rate
        self.ages = ages

    def fit(self, X, y):
        for _ in range(np.shape(X)[1]):
            self.b.append(0)

        ones = np.ones(np.shape(X)[0])
        _X = np.c_[ones, X]

        for _ in range(self.ages):
            e = (y - (_X @ self.b))
            self.b += self.learning_rate * \
                (1 / np.shape(_X)[0]) * \
                np.sum((_X.T * e).T, axis=0)

    def predict(self, x):
        ones = np.ones(np.shape(x)[0])
        _x = np.c_[ones, x]

        return _x @ self.b


class MultipleLinearRegressionSGD():
    def __init__(self, learning_rate=0.01, ages=10):
        self.b = [0]
        self.learning_rate = learning_rate
        self.ages = ages

    def fit(self, X, y):
        for _ in range(np.shape(X)[1]):
            self.b.append(0)

        ones = np.ones(np.shape(X)[0])
        _X = np.c_[ones, X]

        for _ in range(self.ages):
            for i, _x in enumerate(_X):
                self.b += self.learning_rate * (y[i] - (_x @ self.b)) * _x

    def predict(self, x):
        ones = np.ones(np.shape(x)[0])
        _x = np.c_[ones, x]

        return _x @ self.b


class RegularizedLinearRegressionGD():
    def __init__(self, learning_rate=0.01, ages=10, regularization_rate=0):
        self.b_0 = 0
        self.b_j = np.array([])
        self.learning_rate = learning_rate
        self.ages = ages
        self.regularization_rate = regularization_rate

    def fit(self, X, y):
        self.b_j = np.zeros(np.shape(X)[1])

        for _ in range(self.ages):
            e = y - (self.b_0 + X @ self.b_j)

            self.b_0 += self.learning_rate * \
                (1 / np.shape(X)[1]) * np.sum(e)
            self.b_j += self.learning_rate * \
                ((1 / np.shape(X)[1]) * np.sum((X.T * e).T, axis=0) -
                 ((self.regularization_rate / np.shape(X)[1]) * self.b_j))

    def predict(self, x):
        return self.b_0 + x @ self.b_j
