import numpy as np

from .base.layer import Layer


class Dense(Layer):
    def __init__(self, output_row):
        # self.output_shape = (output_size,)
        self.output_row = output_row
        # self.input_shape = None

    def forward(self, input):
        self.input = input
        return np.matmul(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        # transpose axes(row with column)
        input_transposed = np.transpose(self.input, axes=(0, 2, 1))
        weights_transposed = np.transpose(self.weights, axes=(0, 2, 1))

        weights_gradient = np.matmul(output_gradient, input_transposed)
        input_gradient = np.matmul(weights_transposed, output_gradient)

        # axis=0 is taken so that sum is calculated along z-axis
        self.weights -= learning_rate / batch_size * np.sum(weights_gradient, axis=0)
        self.bias -= learning_rate / batch_size * np.sum(output_gradient, axis=0)

        return input_gradient

    def _summary(self):
        return f"Dense Layer ({self.input_shape} -> {self.output_shape})"

    def _initialize_input_shape(self, input_shape):

        self.input_shape = input_shape
        (input_row, input_column) = input_shape

        self.output_shape = (self.output_row, input_column)

        self.weights = np.random.randn(input_column, self.output_row, input_row) / (
            input_row**0.5
        )
        self.bias = np.random.randn(input_column, self.output_row, input_column)

        return self.output_shape
