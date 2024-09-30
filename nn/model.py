from . import losses, optimizers


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

    def get_layers(self):
        return self.layers

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer._set_model(self)

    def compile_for(self, input_shape):
        # input shape is reassigned batch_size to get 3d-matrix, for naive optimizer batch_size=1
        input_shape = (self.optimizer.get_batch_size(), *input_shape)
        self.input_shape = input_shape

        for layer in self.get_layers():
            input_shape = layer._initialize_input_shape(input_shape)

    def summary(self):
        return [layer._summary() for layer in self.layers]

    def train(
        self,
        x_train,
        y_train,
        epochs=1000,
        validate_model=None,
    ):

        for e in range(epochs):

            error = self.optimizer.fit(x_train, y_train)

            error /= len(x_train)
            print(f"Epoch: {e}, Error: {error}")

            if validate_model:
                validate_model(self.predict)
