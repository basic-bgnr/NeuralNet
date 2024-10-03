import numpy as np

from .base.activation import Activation
from .base.layer import Layer


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

    def _summary(self):
        return f"Tanh Activation"


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

    def _summary(self):
        return f"Sigmoid Activation"


class Softmax(Layer):
    def forward(self, input):
        rank = np.ndim(input)
        axis = (
            (rank - 2, rank - 1) if rank >= 2 else (0,)
        )  # sum across plane(row, column)
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=axis, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]

        n = np.size(self.output) // batch_size
        input_gradient = np.matmul(
            (np.identity(n) - np.transpose(self.output, axes=(0, 2, 1))) * self.output,
            output_gradient,
        )
        return input_gradient

    def _summary(self):
        return f"Softmax Activation"


class Relu(Activation):
    def __init__(self):
        def relu(x):
            xx = np.copy(x)
            xx[xx < 0] = 0.0
            return xx

        def relu_prime(x):
            xx = relu(x)
            xx[xx > 0] = 1.0
            return xx

        super().__init__(relu, relu_prime)

    def _summary(self):
        return f"Relu Activation"
