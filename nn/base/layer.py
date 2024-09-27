class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

    def _summary(self):
        # TODO: generate summary
        pass

    def _initialize_input_shape(self, input_shape):
        """
        Initialize internal matrix representation using input_shape.
        args:
        input_shape: list of input_matrix dimension [depth, width, height]
        Returns: (output_shape: which is the input_shape for the next layer)
        1.Default case: returns the input_shape as it is [for activation layer
        such as sigmoid, tanh, softmax, relu, the input_shape is exactly equal
        to the output_shape]
        """
        return input_shape
