from . import losses, optimizers
from .dropout import Dropout


class Model:
    def __init__(self):
        self.layers = None
        self.input_shape = None

    # public function
    def predict(self, input):
        input = input.reshape(1, *input.shape)  # add extra dimension for batch_size
        for layer in filter(lambda layer: not isinstance(layer, Dropout), self.layers):
            input = layer.forward(input)
        return input

    # internal function
    def forward(self, input):
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
        # input_shape = (self.optimizer.get_batch_size(), *input_shape)
        self.input_shape = input_shape

        for layer in self.get_layers():
            input_shape = layer._initialize_input_shape(input_shape)

    def summary(self):
        return [layer._summary() for layer in self.layers]

    def train(self, x_train, y_train, epochs=1000, validate_model=None, verbose=False):

        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")

            error = self.optimizer.fit(x_train, y_train, verbose=verbose)

            error /= len(x_train)

            print(f"Error: {error}")
            if validate_model:
                validate_model(self.predict)

            # print a newlint at the end of epoch
            print()
