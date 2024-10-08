import itertools

import numpy as np
import tqdm


class _Optimizer:
    def _set_model(self, model):
        self.model = model

    def get_batch_size(self):
        pass

    def fit(self, x_train, y_train, verbose=False):
        pass


class Naive(_Optimizer):
    def __init__(self, learning_rate, cost_function):
        self.learning_rate = learning_rate
        self.cost_function = cost_function

    def get_batch_size(self):
        return 1

    def fit(self, x_train, y_train, verbose=False):
        epoch_error = 0.0
        x_y_train = zip(x_train, y_train)
        x_y_train = tqdm.tqdm(x_y_train) if verbose else x_y_train
        for x, y in x_y_train:
            # forward,
            output = self.model.forward(x.reshape(self.model.input_shape))

            # error
            epoch_error += self.cost_function.loss(y, output)

            # backward
            grad = self.cost_function.loss_prime(y, output)
            for layer in reversed(self.model.get_layers()):
                grad = layer.backward(grad, self.learning_rate, self.get_batch_size())

        return np.sum(epoch_error)


class SGD(_Optimizer):
    def __init__(self, batch_size, learning_rate, cost_function):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_function = cost_function

    def get_batch_size(self):
        return self.batch_size

    def _batchify(self, x_train, y_train):
        random_indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[random_indices], y_train[random_indices]
        for start in range(0, x_train.shape[0], self.batch_size):
            if start + self.batch_size <= len(x_train):  # discard irregular batch
                yield x_train[start : start + self.batch_size], y_train[
                    start : start + self.batch_size
                ]

    def fit(self, x_train, y_train, verbose=False):
        epoch_error = 0

        batches = self._batchify(x_train, y_train)
        batches = (
            tqdm.tqdm(
                batches,
                desc="Processing mini-batches",
                total=len(x_train) // self.get_batch_size(),
            )
            if verbose
            else batches
        )

        for x_batch, y_batch in batches:

            output = self.model.forward(x_batch)

            # error
            epoch_error += self.cost_function.loss(y_batch, output)

            # backward
            gradient = self.cost_function.loss_prime(y_batch, output)
            for layer in reversed(self.model.get_layers()):
                gradient = layer.backward(gradient, self.learning_rate)

            # update parameter at the end of batch
        return np.sum(epoch_error)
