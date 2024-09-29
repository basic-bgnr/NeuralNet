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

    def _initialize_cumulative_gradient(self):
        """
        This is needed to implement SGD
        Initialize cumulative gradient for weight, bais, kernel matrix (if required)
        Layer that does required to set cum gradient for need to implement this explicitly.
        For Activation layer this should not do anything
        """
        pass

    def _backward_sgd(self, output_gradient):
        """
        This is needed to implement SGD
        note: THIS FUNCTION RETURNS THE INPUT_GRADIENT.
        This function takes the output_gradient from n+1 layer and calcultes
        the current layer's parameter's gradient(if needed).
        (if needed) The gradient is then added with internal variable that tracks
        the cumulative sum of the gradient for the entire batch
        Activation layer doesn't need to update their parameters (since they are parameterless)
        """
        return self.backward(output_gradient, learning_rate=None)

    def _update_parameter_sgd(self, learning_rate, batch_size):
        """
        This is needed to implement SGD
        This function is called at end of batch process. This normalizes the cumulative
        parameter with batch size and add that cumulative sum to parameter after scaling
        with learning rate.
        Activation layer doesn't need to implement this function as they don't have any
        internal parameters
        For any layer that overrides this implementation,
        Remember: to call _init_cum_gradient to set cum_grad parameters to zero at the end
        of batch
        """
        pass
