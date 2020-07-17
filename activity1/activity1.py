import numpy as np

from metrics import RSS, RSE, R2, MAE


class SimpleLinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.b_1 = np.sum((X - np.mean(X)) * (y - np.mean(y))) / \
            np.sum(np.power((X - np.mean(X)), 2))
        self.b_0 = np.mean(y) - self.b_1 * np.mean(X)

    def predict(self, x):
        return self.b_0 + self.b_1 * np.array(x)


class MultipleLinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        ones = np.ones(np.shape(X)[0])
        X = np.c_[ones, X]

        self.b = np.linalg.pinv(np.transpose(X) @ X) @ np.transpose(X) @ y

    def predict(self, x):
        ones = np.ones(np.shape(x)[0])
        x = np.c_[ones, x]

        return x @ self.b


def test_simple_linear_regression():
    print('----------')
    print('Simple Linear Regression', '\n')

    X_train, y_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    X_test, y_test = [11, 12, 13, 14, 15], [23, 25, 27, 29, 31]

    model = SimpleLinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print('X test:', X_test, ' | ', 'y predict', y_predict, '\n')

    print('RSS:', RSS(y_test, y_predict))
    print('RSE:', RSE(y_test, y_predict))
    print('R2:', R2(y_test, y_predict))
    print('MAE:', MAE(y_test, y_predict))

    print('----------')


def test_multiple_linear_regression():
    print('----------')
    print('Simple Linear Regression', '\n')

    X_train, y_train = [[1, 2, 3], [4, 5, 6],
                        [7, 8, 9], [10, 11, 12]], [3, 5, 7, 9]
    X_test, y_test = [[11, 12, 13], [14, 15, 16], [17, 18, 19]], [10, 12, 14]

    model = MultipleLinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print('X test:', X_test, ' | ', 'y predict', y_predict, '\n')

    print('RSS:', RSS(y_test, y_predict))
    print('RSE:', RSE(y_test, y_predict))
    print('R2:', R2(y_test, y_predict))
    print('MAE:', MAE(y_test, y_predict))

    print('----------')


def main():
    test_simple_linear_regression()
    test_multiple_linear_regression()


if __name__ == "__main__":
    main()
