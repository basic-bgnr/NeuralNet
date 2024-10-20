from enum import Enum

import numpy as np
from scipy import signal

from .base.layer import Layer


class ConvolutionalMode(Enum):
    Valid = "valid"
    Same = "same"


class Convolutional(Layer):

    def __init__(self, kernel_size, depth, mode=ConvolutionalMode.Valid):
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

        self.mode = mode

    def forward(self, input):
        batch_size = input.shape[0]

        match self.mode:
            case ConvolutionalMode.Valid:
                self.input = input
            case ConvolutionalMode.Same:
                height_pad, width_pad = (
                    (self.kernel_size - 1) // 2,
                    (self.kernel_size - 1) // 2,
                )
                self.input = np.pad(
                    input,
                    ((0, 0), (0, 0), (height_pad, height_pad), (width_pad, width_pad)),
                    mode="constant",
                )

        self.output = np.zeros((batch_size, *self.output_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                self.output[b, i] = signal.correlate(
                    self.input[b], self.kernels[i], "valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]

        kernels_gradient = np.zeros((batch_size, *self.kernels_shape))
        bias_gradient = output_gradient
        input_gradient = np.zeros((batch_size, *self.input_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    kernels_gradient[b, i, j] = signal.correlate(
                        self.input[b, j], output_gradient[b, i], "valid"
                    )
                    input_gradient[b, j] = signal.convolve(
                        output_gradient[b, i], self.kernels[i, j], "full"
                    )

        self.kernels -= learning_rate * np.sum(kernels_gradient, axis=0)
        self.bias -= learning_rate * np.sum(bias_gradient, axis=0)

        match self.mode:
            case ConvolutionalMode.Valid:
                return input_gradient
            case ConvolutionalMode.Same:
                _, height, width = self.input_shape
                height_pad, width_pad = (
                    (self.kernel_size - 1) // 2,
                    (self.kernel_size - 1) // 2,
                )
                return input_gradient[
                    :,
                    :,
                    height_pad : (height - height_pad),
                    height_pad : (width - width_pad),
                ]

    def _summary(self):
        return f"Convolution Layer {self.input_shape} -> {self.output_shape}"

    def _initialize_input_shape(self, input_shape):

        match self.mode:
            case ConvolutionalMode.Valid:
                (input_depth, input_height, input_width) = input_shape

                self.input_depth = input_depth
                self.input_shape = (input_depth, input_height, input_width)

            case ConvolutionalMode.Same:
                (input_depth, input_height, input_width) = input_shape

                input_height = input_height + self.kernel_size - 1
                input_width = input_width + self.kernel_size - 1

                self.input_depth = input_depth
                self.input_shape = (input_depth, input_height, input_width)

        bias_shape = (
            self.depth,
            input_height - self.kernel_size + 1,
            input_width - self.kernel_size + 1,
        )
        self.output_shape = bias_shape
        self.kernels_shape = (
            self.depth,
            input_depth,
            self.kernel_size,
            self.kernel_size,
        )

        self.kernels = np.random.randn(*self.kernels_shape) / (self.kernel_size**0.5)
        self.bias = np.random.randn(*bias_shape)

        return self.output_shape
