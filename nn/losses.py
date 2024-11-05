import numpy as np


class MSE:
    def loss(self, y_true, y_pred):
        """
        Here axis=(rank-2, rank-1) is taken so that sum is calculated across x,y plane (works for both 3d/2d)
        """
        rank = np.ndim(y_true)
        axis = tuple(
            range(1, rank)
        )  # sum across all axis except the first (batch dimension)
        return np.sum(np.power(y_true - y_pred, 2), axis=axis)

    def loss_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


class CrossEntropy:
    def loss(self, y_true, y_pred):
        """
        Here axis=(rank-2, rank-1) is taken so that sum is calculated across x,y plane (works for both 3d/2d)
        """
        rank = np.ndim(y_true)
        axis = tuple(
            range(1, rank)
        )  # sum across all axis except the first (batch dimension)
        return -np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=axis
        )

    def loss_prime(self, y_true, y_pred):
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
