from builtins import object, range

import numpy as np
from layer_utils import affine_relu_backward, affine_relu_forward
from layers import affine_backward, affine_forward, softmax_loss


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params["W1"] = np.random.normal(
            scale=weight_scale, size=(input_dim, hidden_dim)
        )
        self.params["b1"] = np.zeros(hidden_dim)

        self.params["W2"] = np.random.normal(
            scale=weight_scale, size=(hidden_dim, num_classes)
        )
        self.params["b2"] = np.zeros(num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # affine - relu
        A1, layer1_cache = affine_relu_forward(
            X, self.params["W1"], self.params["b1"]
        )
        # affine
        Z2, fc2_cache = affine_forward(A1, self.params["W2"], self.params["b2"])
        scores = Z2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx_fin = softmax_loss(Z2, y)
        loss += (
            0.5
            * self.reg
            * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        )
        dx2, grads["W2"], grads["b2"] = affine_backward(dx_fin, fc2_cache)
        grads["W2"] += self.reg * self.params["W2"]

        _, grads["W1"], grads["b1"] = affine_relu_backward(dx2, layer1_cache)
        grads["W1"] += self.reg * self.params["W1"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads
