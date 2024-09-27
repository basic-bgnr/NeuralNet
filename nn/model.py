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
        loss,
        x_train,
        y_train,
        epochs=1000,
        learning_rate=0.01,
        validate_model=None,
    ):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss.loss(y, output)

                # backward
                grad = loss.loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            print(f"Epoch: {e}, Error: {error}")
            if validate_model:
                validate_model(self.predict)
