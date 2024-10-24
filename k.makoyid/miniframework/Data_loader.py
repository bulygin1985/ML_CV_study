#DataLoader.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


class DataLoader:
    @staticmethod
    def load_cifar10_data():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

        y_train = tf.keras.utils.to_categorical(y_train.squeeze(), 10).T
        y_test = tf.keras.utils.to_categorical(y_test.squeeze(), 10).T

        return x_train, y_train, x_test, y_test