import numpy as np
from Activation_Functions import ActivationFunctions


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, num_classes):
        self.W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(num_classes, hidden_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((num_classes, 1))

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = ActivationFunctions.leaky_relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = ActivationFunctions.softmax(Z2)

        cache = (Z1, A1, Z2, A2)
        return A2, cache

    def backward_propagation(self, X, Y, cache, lambda_wd):
        m = X.shape[1]
        (Z1, A1, Z2, A2) = cache

        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T) + (lambda_wd / m) * self.W2
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = ActivationFunctions.leaky_relu_backward(dA1, Z1)
        dW1 = 1 / m * np.dot(dZ1, X.T) + (lambda_wd / m) * self.W1
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, gradients, learning_rate):
        self.W1 -= learning_rate * gradients["dW1"]
        self.b1 -= learning_rate * gradients["db1"]
        self.W2 -= learning_rate * gradients["dW2"]
        self.b2 -= learning_rate * gradients["db2"]