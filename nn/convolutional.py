import numpy as np
from scipy import signal

from .base.layer import Layer


class Convolutional(Layer):

    def __init__(self, kernel_size, depth):
        """
        Partially initializes convolutional layer.
        Full initialization is done my Model class after shape of input layer
        is finalized.
        """
        self.kernel_size = kernel_size
        self.depth = depth

        self.input_depth = None
        self.input_shape = None
        self.output_shape = None
        self.kernel_shape = None

        self.kernels = None
        self.bias = None

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    def _summary(self):
        return "Convolution Layer {self.input_shape} -> {self.output_shape}"

    def _initialize_input_shape(self, input_shape):
        (input_depth, input_height, input_width) = input_shape

        self.input_depth = input_depth
        self.input_shape = input_shape
        self.output_shape = (
            self.depth,
            input_height - self.kernel_size + 1,
            input_width - self.kernel_size + 1,
        )
        self.kernels_shape = (
            self.depth,
            input_depth,
            self.kernel_size,
            self.kernel_size,
        )

        self.kernels = np.random.randn(*self.kernels_shape) / (self.kernel_size**0.5)
        self.biases = np.random.randn(*self.output_shape)

        return self.output_shape
