import gzip
import pathlib
import struct

import numpy as np


def _load_raw_data(
    image_file_location,
    label_file_location,
):
    read_integer = lambda file_object: struct.unpack(">i", file_object.read(4))[0]
    read_image = lambda file_object, row, column: file_object.read(row * column)
    read_byte = lambda file_object: struct.unpack("B", file_object.read(1))[0]

    datas = []

    with (
        gzip.open(image_file_location, "rb") as training_images_file,
        gzip.open(label_file_location, "rb") as training_labels_file,
    ):
        magic_number_images = read_integer(training_images_file)
        magic_number_labels = read_integer(training_labels_file)

        number_images = read_integer(training_images_file)
        number_labels = read_integer(training_labels_file)

        image_row = read_integer(training_images_file)
        image_col = read_integer(training_images_file)

        assert magic_number_images == 2051, magic_number_labels == 2049
        assert number_images == number_labels
        assert image_row == image_col, image_row == 32

        images, labels = (
            np.empty(shape=(number_images, image_row, image_col), dtype=np.uint8),
            np.empty(shape=(number_labels,), dtype=np.uint8),
        )

        for i in range(number_images):
            image_data = read_image(training_images_file, image_row, image_col)
            image_label = read_byte(training_labels_file)

            images[i, :, :] = np.frombuffer(image_data, np.uint8).reshape((28, 28))
            labels[i] = image_label

    return images, labels


def load_data():
    data_set_location = f"{pathlib.Path(__file__).parent}/mnist_dataset"

    training_image_location = f"{data_set_location}/train-images-idx3-ubyte.gz"
    training_label_location = f"{data_set_location}/train-labels-idx1-ubyte.gz"

    testing_image_location = f"{data_set_location}/t10k-images-idx3-ubyte.gz"
    testing_label_location = f"{data_set_location}/t10k-labels-idx1-ubyte.gz"

    (x_train, y_train) = _load_raw_data(
        training_image_location, training_label_location
    )
    (x_test, y_test) = _load_raw_data(testing_image_location, testing_label_location)

    return (x_train, y_train), (x_test, y_test)
