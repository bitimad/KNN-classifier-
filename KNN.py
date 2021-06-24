import numpy as np
from scipy.stats import mode


class KNN(object):
    def __init__(self, distance, k, q=0):
        """ distance: the type of distance to be used
            k: number of neighbors
            q:  used for minkowski distance metric only, so for other distances 0"""
        self.distance = distance
        self.k = k
        self.q = q

    @staticmethod
    def euclidean(x, y):
        return np.sqrt(np.sum(np.square(x - y), axis=1))

    @staticmethod
    def manhatten(x, y):
        return np.sum(np.abs(x - y), axis=1)

    @staticmethod
    def minkowski(x, y, q):
        return np.power(np.sum(np.power(np.abs(x - y), q), axis=1), 1 / q)

    def predict(self, X_test, X_train, y_train):
        assert X_test.shape == (1, X_train.shape[1])
        if self.distance == 'euclidean':
            y = self.euclidean(X_train, X_test)
        elif self.distance == 'manhatten':
            y = self.manhatten(X_train, X_test)
        elif self.distance == 'minkowski':
            y = self.minkowski(X_train, X_test, self.q)
        else:
            print("Wrong Distance metric!!")
            return None

        ind = y.argsort()[:self.k]
        return mode(y_train[ind]).mode[0]

    def accuracy(self, X_test, X_train, y_train, y_test):
        predictions = []
        for sample in X_test:
            sample = np.expand_dims(sample, axis=0)
            predictions.append(self.predict(sample, X_train, y_train))
        predictions = np.array(predictions)
        print(f"Accuracy : {(predictions == y_test).sum() / len(y_test):.4f}")
