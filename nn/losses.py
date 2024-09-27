import numpy as np


class MSE:
    @classmethod
    def loss(cls, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @classmethod
    def loss_prime(cls, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)


class BinaryCrossEntropy:
    @classmethod
    def loss(cls, y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    @classmethod
    def loss_prime(cls, y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
