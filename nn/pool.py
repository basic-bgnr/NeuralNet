import numpy as np

from .base.layer import Layer


class Pool(Layer):
    def __init__(self, size, forward_pool_func, backward_pool_func):
        self.size = size
        # eg. pool_func can be np.max, np.mean or user defined
        self.forward_pool_func = forward_pool_func
        self.backward_pool_func = backward_pool_func

    def forward(self, input):
        self.input = input
        return self._get_output(input)

    def backward(self, output_gradient, learning_rate):
        return self._generate_backward_mask() * self._resize_to_input_shape(
            output_gradient
        )

    def _generate_backward_mask(self):
        batch_size, channel, row, column = self.input.shape
        mask = np.zeros((batch_size, channel, row, column))

        for r in range(0, row, self.size):
            for c in range(0, column, self.size):
                kernel_volume = self.input[:, :, r : r + self.size, c : c + self.size]
                kernel_volume = kernel_volume.reshape(
                    (
                        batch_size,
                        channel,
                        kernel_volume.shape[-1] * kernel_volume.shape[-2],
                    )
                )
                max_index = self.backward_pool_func(kernel_volume, axis=2)
                max_row_index, max_column_index = (
                    r + max_index // self.size,
                    c + max_index % self.size,
                )

                max_row_index = max_row_index.squeeze()
                max_column_index = max_column_index.squeeze()
                max_channel_index = np.tile(np.arange(channel), batch_size)
                max_batch_index = np.repeat(np.arange(batch_size), channel)

                mask[
                    (
                        max_batch_index,
                        max_channel_index,
                        max_row_index,
                        max_column_index,
                    )
                ] = 1

        return mask

    def _reshape_to_kernel_size(self, input):
        batch_size, channel, R, C = input.shape
        r, c = R // self.size, C // self.size
        return input.reshape((batch_size, channel, r, self.size, c, self.size))

    def _get_output(self, input):
        input = self._reshape_to_kernel_size(input)
        return self.forward_pool_func(input, axis=(3, 5))

    def _resize_to_input_shape(self, tensor):
        return tensor.repeat(self.size, axis=2).repeat(self.size, axis=3)

    def _initialize_input_shape(self, input_shape):
        (channel, row, column) = input_shape
        return (channel, row // self.size, column // self.size)


class MaxPool(Pool):
    def __init__(self, size):
        super().__init__(size, np.max, np.argmax)

    def _summary(self):
        return f"Max Layer ({self.size})"


class MinPool(Pool):
    def __init__(self, size):
        super().__init__(size, np.min, np.argmin)

    def _summary(self):
        return f"MinPool Layer ({self.size})"


class AvgPool(Pool):
    def __init__(self, size):
        super().__init__(size, np.mean, None)

    def backward(self, output_gradient, learning_rate):
        return self._resize_to_input_shape(output_gradient)

    def _summary(self):
        return f"AvgPool Layer ({self.size})"
