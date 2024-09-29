from . import losses
from . import optimizers


class Model:
    def __init__(self):
        self.layers = None
        self.input_shape = None

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def set_layers(self, layers):
        self.layers = layers

    def compile_for(self, input_shape):
        self.input_shape = input_shape
        for layer in self.layers:
            input_shape = layer._initialize_input_shape(input_shape)

    def summary(self):
        return [layer._summary() for layer in self.layers]

    def train(
        self,
        x_train,
        y_train,
        optimizer = optimizers.Naive(cost_function=losses.MSE(), learning_rate=0.01),
        epochs=1000,
        validate_model=None,
    ):
        optimizer._set_model(self)

        for e in range(epochs):

            error = optimizer.fit(x_train, y_train)

            error /= len(x_train)
            print(f"Epoch: {e}, Error: {error}")

            if validate_model:
                validate_model(self.predict)
