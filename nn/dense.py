import numpy as np

from .base.layer import Layer


class Dense(Layer):
    def __init__(self, output_size):
        self.output_shape = (output_size,)
        self.input_shape = None

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

    def _summary(self):
        return f"Dense Layer ({self.input_shape} -> {self.output_shape})"

    def _initialize_input_shape(self, input_shape):

        self.input_shape = input_shape

        input_size = self.input_shape[0]
        output_size = self.output_shape[0]

        self.weights = np.random.randn(output_size, input_size) / (input_size**0.5)
        self.bias = np.random.randn(output_size, 1)

        return self.output_shape
