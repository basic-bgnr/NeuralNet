class Model:
    def __init__(self):
        self.layers = None

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def set_layers(self, layers):
        self.layers = layers

    def train(
        self,
        loss,
        loss_prime,
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
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            print(f"Epoch: {e}, Error: {error}")
            if validate_model:
                validate_model(self.predict)
