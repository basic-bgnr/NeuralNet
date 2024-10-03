import numpy as np

from .base.layer import Layer


class Reshape(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape
        self.input_shape = None

    def forward(self, input):
        batch_size = input.shape[0]
        return np.reshape(input, (batch_size, *self.output_shape))

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        return np.reshape(output_gradient, (batch_size, *self.input_shape))

    def _summary(self):
        return f"Reshape Layer ({self.input_shape}) -> {self.output_shape}"

    def _initialize_input_shape(self, input_shape):
        self.input_shape = input_shape
        return self.output_shape


class Flatten(Reshape):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def _initialize_input_shape(self, input_shape):
        (depth, row, column) = input_shape
        self.input_shape = input_shape

        self.output_shape = (depth * row * column, 1)
        return self.output_shape
