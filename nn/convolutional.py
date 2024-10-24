import time
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

        output = np.zeros((batch_size, *self.output_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                output[b, i] = signal.correlate(self.input[b], self.kernels[i], "valid")
        return output + self.bias

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]

        kernels_gradient = np.zeros((batch_size, *self.kernels_shape))
        bias_gradient = output_gradient
        input_gradient = np.zeros((batch_size, *self.input_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    kernels_gradient[b, i, j] = signal.correlate2d(
                        self.input[b, j], output_gradient[b, i], "valid"
                    )
                    input_gradient[b, j] += signal.convolve2d(
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
                    height_pad:-height_pad,
                    height_pad:-width_pad,
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

        rng = np.random.default_rng(seed=time.time_ns())
        self.kernels = rng.standard_normal(self.kernels_shape) / (self.kernel_size**0.5)
        self.bias = rng.standard_normal(bias_shape)

        return self.output_shape


class FastConvolutional(Layer):

    def __init__(self, kernel_size, depth, mode=ConvolutionalMode.Valid, stride=1):
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
        self.stride = 1

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

        _, _, filter_height, filter_width = self.kernels_shape

        output_features, output_height, output_width = self.output_shape

        output = np.zeros((batch_size, output_features, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                start_y = i * self.stride
                end_y = start_y + filter_height
                start_x = j * self.stride
                end_x = start_x + filter_width

                input_patches = self.input[:, :, start_y:end_y, start_x:end_x]
                output[:, :, i, j] = np.einsum(
                    "bcij,fcij->bf", input_patches, self.kernels
                )

        return output + self.bias

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]

        _, input_height, input_width = self.input_shape
        _, _, filter_height, filter_width = self.kernels_shape

        # # Create a padded version of the gradient for full convolution
        output_gradient_padded = np.pad(
            output_gradient,
            (
                (0, 0),
                (0, 0),
                (filter_height - 1, filter_height - 1),
                (filter_width - 1, filter_width - 1),
            ),
            "constant",
        )
        # Calculate the gradient of the loss with respect to the input of the layer
        input_gradient = np.zeros((batch_size, *self.input_shape))
        # flip to perform convolution
        rot_kernels = np.flip(self.kernels, axis=(2, 3))
        for i in range(input_height):
            for j in range(input_width):
                start_y = i * self.stride
                end_y = start_y + filter_height
                start_x = j * self.stride
                end_x = start_x + filter_width

                input_gradient[:, :, i, j] = np.einsum(
                    "bfij,fcij->bc",
                    output_gradient_padded[
                        :,
                        :,
                        start_y:end_y,
                        start_x:end_x,
                    ],
                    rot_kernels,
                )

        # Calculate the gradients of the weights and biases
        kernels_gradient = np.zeros((batch_size, *self.kernels_shape))
        bias_gradient = output_gradient

        # perfrom cross-correlation between input and output_gradient
        _, output_gradient_height, output_gradient_width = self.output_shape
        for i in range(filter_height):
            for j in range(filter_width):
                start_y = i
                end_y = start_y + output_gradient_height
                start_x = j
                end_x = start_x + output_gradient_width

                kernels_gradient[:, :, :, i, j] = np.einsum(
                    "bcij, bfij -> bfc",
                    self.input[:, :, start_y:end_y, start_x:end_x],
                    output_gradient,
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
                    height_pad:-height_pad,
                    height_pad:-width_pad,
                ]

    def _summary(self):
        return f"FastConvolution Layer {self.input_shape} -> {self.output_shape}"

    def _initialize_input_shape(self, input_shape):

        (input_depth, input_height, input_width) = input_shape

        self.input_depth = input_depth
        self.input_shape = (input_depth, input_height, input_width)

        match self.mode:
            case ConvolutionalMode.Valid:
                (input_depth, input_height, input_width) = input_shape

                self.input_depth = input_depth
                self.input_shape = (input_depth, input_height, input_width)
                # self.padding = 0

            case ConvolutionalMode.Same:
                (input_depth, input_height, input_width) = input_shape

                input_height = input_height + self.kernel_size - 1
                input_width = input_width + self.kernel_size - 1

                self.input_depth = input_depth
                self.input_shape = (input_depth, input_height, input_width)
                # self.padding = (self.kernel_size - 1)//2

        # match self.mode:
        #     case ConvolutionalMode.Valid:
        #         self.padding = 0

        #     case ConvolutionalMode.Same:
        #         self.padding = (self.kernel_size - 1) // 2

        bias_shape = (
            self.depth,
            (input_height - self.kernel_size + 1) // self.stride,
            (input_width - self.kernel_size + 1) // self.stride,
        )

        self.output_shape = bias_shape
        self.kernels_shape = (
            self.depth,
            input_depth,
            self.kernel_size,
            self.kernel_size,
        )

        rng = np.random.default_rng(seed=time.time_ns())
        self.kernels = rng.standard_normal(self.kernels_shape) / (self.kernel_size**0.5)
        self.bias = rng.standard_normal(bias_shape)

        return self.output_shape
