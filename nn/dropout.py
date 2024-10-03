import numpy as np

from .base.layer import Layer


class Dropout(Layer):
    def __init__(self, dropout_ratio):
        self.drop_out_ratio = dropout_ratio

    def forward(self, input):
        self.drop_out = (
            np.random.uniform(low=0.0, high=1.0, size=input.shape) > self.drop_out_ratio
        )
        self.drop_out = self.drop_out.astype(np.float64)
        return self.drop_out * input

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.drop_out

    def _summary(self):
        return f"Dropout Layer ({self.drop_out_ratio})%"
