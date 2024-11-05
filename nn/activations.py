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
        axis = tuple(
            range(1, rank)
        )  # sum across all axis except the first (batch dimension)
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=axis, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        _, width, height = output_gradient.shape

        n = width * height
        input_gradient = np.matmul(
            (np.identity(n) - np.transpose(self.output, axes=(0, 2, 1))) * self.output,
            output_gradient,
        )
        return input_gradient

    def _summary(self):
        return f"Softmax Activation"


class Softmax2d(Layer):
    """This class works only when summation axis = z, (see overfeat paper and
    overfeat_multidigit_lenet5.ipynb for details)"""
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=self.axis, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size, channel, width, height = output_gradient.shape
        input_gradient = np.zeros_like(output_gradient)
        for b in range(batch_size):
            for i in range(channel):
                for j in range(width):
                    for k in range(height):
                        identity = np.zeros(channel)
                        identity[i] = 1.0
                        output_strip = self.output[b, :, j, k]
                        gradient_strip = output_gradient[b, :, j, k]
                        input_gradient_element = np.sum(
                            gradient_strip
                            * self.output[b, i, j, k]
                            * (identity - output_strip)
                        )
                        input_gradient[b, i, j, k] = input_gradient_element

        return input_gradient

    def _summary(self):
        return f"Softmax2d Activation"


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
