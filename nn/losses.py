import numpy as np


class MSE:
    def loss(self, y_true, y_pred):
        """
        Here axis=1 is taken so that sum is calculated across row
        """
        return np.sum(np.power(y_true - y_pred, 2), axis=1)

    def loss_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


class CrossEntropy:
    def loss(self, y_true, y_pred):
        """
        Here axis=1 is taken so that sum is calculated across row
        """
        return -np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1
        )

    def loss_prime(self, y_true, y_pred):
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
