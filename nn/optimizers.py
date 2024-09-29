import itertools

import numpy as np


class Naive:
    def __init__(self, learning_rate, cost_function):
        self.learning_rate = learning_rate
        self.cost_function = cost_function
    
    def _set_model(self, model):
        self.model = model
        for layer in self.model.layers:
            layer._initialize_cumulative_gradient()
    
    def fit(self, x_train, y_train):
        epoch_error = 0.0
        for x, y in zip(x_train, y_train):
            # forward
            output = self.model.predict(x)

            # error
            epoch_error += self.cost_function.loss(y, output)

            # backward
            grad = self.cost_function.loss_prime(y, output)
            for layer in reversed(self.model.layers):
                grad = layer.backward(grad, self.learning_rate)

        return epoch_error
            
class SGD:
    def __init__(self, batch_size, learning_rate, cost_function):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_function = cost_function

        # This initializes the layers to have additional cum_grad_ parameters
        # so that we can apply SGD"
    def _set_model(self, model):
        self.model = model
        for layer in self.model.layers:
            layer._initialize_cumulative_gradient()

    def _batchify(self, x_train, y_train):
        random_indices = np.random.permutation(len(x_train))
        training_data = zip(x_train[random_indices], y_train[random_indices])

        return itertools.batched(training_data, self.batch_size)

    def fit(self, x_train, y_train):
        epoch_error = 0
        for batch in self._batchify(x_train, y_train):
            for x, y in batch:

                output = self.model.predict(x)

                # error
                epoch_error += self.cost_function.loss(y, output)

                # backward
                gradient = self.cost_function.loss_prime(y, output)
                for layer in reversed(self.model.layers):
                    gradient = layer._backward_sgd(gradient)

            # update parameter at the end of batch 
            self._normalize_and_update_batch()
        return epoch_error

    def _normalize_and_update_batch(self):
        for layer in self.model.layers:
            layer._update_parameter_sgd(self.learning_rate, self.batch_size)
