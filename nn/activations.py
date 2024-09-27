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
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

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
