import numpy as np

from .base.layer import Layer


class Reshape(Layer):
    def __init__(self, output_shape):
        self.output_shape = output_shape
        self.input_shape = None

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

    def _summary(self):
        return "Reshape Layer ({self.input_shape}) -> {self.output_shape}"

    def _initialize_input_shape(self, input_shape):
        self.input_shape = input_shape
        return self.output_shape


class Flatten(Reshape):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def _initialize_input_shape(self, input_shape):
        self.input_shape = input_shape

        output_shape = 1
        for s in self.input_shape:
            output_shape *= s
        self.output_shape = (output_shape, 1)
        return self.output_shape
