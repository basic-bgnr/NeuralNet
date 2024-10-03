import numpy as np

from .base.layer import Layer


class Pool(Layer):
    def __init__(self, size, pool_func):
        self.size = size
        #eg. pool_func can be np.max, np.mean or user defined
        self.pool_func = pool_func

    def forward(self, input):
        return self._get_output(input)

    def backward(self, output_gradient, learning_rate):
        pass

    def _reshape_to_kernel_size(self, input):
        batch_size, R, C = input.shape
        r, c = R // self.size, C // self.size
        return input.reshape((batch_size, r, self.size, c, self.size))

    def _get_output(self, input):
        input = self._reshape_to_kernel_size(input)
        return self.pool_func(input, axis=(2, 4))

    def _initialize_input_shape(self, input_shape):
        _, row, column = input_shape
        return row // self.size, column // self.size


class MaxPool(Pool):
    def __init__(self, size):
        super().__init__(size, np.max)

    def backward(self, output_gradient, learning_rate):
        pass

    def _summary(self):
        return f"Max Layer ({self.drop_out_ratio})%"


class MeanPool(Pool):
    def __init__(self, size):
        super().__init__(size, np.mean)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.repeat(self.size, axis=1).repeat(self.size, axis=2)

    def _summary(self):
        return f"MeanPool Layer ({self.drop_out_ratio})%"
