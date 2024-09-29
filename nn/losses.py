import numpy as np


class MSE:
    def loss(self, y_true, y_pred):
        return np.sum(np.power(y_true - y_pred, 2))

    def loss_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


class CrossEntropy:
    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def loss_prime(self, y_true, y_pred):
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
