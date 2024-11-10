#BatchNorm2d
import numpy as np


class BatchNorm2d:
    """
    2D Batch Normalization layer implementation for convolutional networks
    """

    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Initialize BatchNorm2d layer

        Args:
            num_features (int): Number of input channels
            epsilon (float): Small constant for numerical stability
            momentum (float): Momentum for running mean/variance computation
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters - shape (1, C, 1, 1) for proper broadcasting
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))

        # Running estimates for inference
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        # Cache for backward pass
        self.cache = None
        self.training = True

    def forward(self, x, training=True):
        """
        Forward pass of 2D batch normalization

        Args:
            x: Input data of shape (N, C, H, W)
            training (bool): Whether in training or inference mode

        Returns:
            out: Normalized, scaled and shifted data
        """
        self.training = training
        N, C, H, W = x.shape

        if training:
            # Calculate mean and variance across batch, height, and width dimensions
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Scale and shift
            out = self.gamma * x_normalized + self.beta

            # Update running estimates
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Store cache for backward pass
            self.cache = {
                'x': x,
                'normalized': x_normalized,
                'mean': batch_mean,
                'var': batch_var,
                'sqrt_var': np.sqrt(batch_var + self.epsilon)
            }

        else:
            # Use running estimates in inference mode
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        """
        Backward pass of 2D batch normalization

        Args:
            dout: Upstream derivatives of shape (N, C, H, W)

        Returns:
            dx: Gradient with respect to input x
            dgamma: Gradient with respect to scale parameter gamma
            dbeta: Gradient with respect to shift parameter beta
        """
        if not self.training:
            raise RuntimeError("Backward pass called during inference")

        x = self.cache['x']
        normalized = self.cache['normalized']
        mean = self.cache['mean']
        var = self.cache['var']
        sqrt_var = self.cache['sqrt_var']
        N, C, H, W = x.shape
        M = N * H * W  # Effective batch size

        # Gradient with respect to beta
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        # Gradient with respect to gamma
        dgamma = np.sum(dout * normalized, axis=(0, 2, 3), keepdims=True)

        # Gradient with respect to normalized x
        dx_normalized = dout * self.gamma

        # Gradient with respect to variance
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (var + self.epsilon) ** (-1.5),
                     axis=(0, 2, 3), keepdims=True)

        # Gradient with respect to mean
        dmean = np.sum(dx_normalized * -1 / sqrt_var, axis=(0, 2, 3), keepdims=True) + \
                dvar * np.mean(-2 * (x - mean), axis=(0, 2, 3), keepdims=True)

        # Gradient with respect to input x
        dx = dx_normalized / sqrt_var + \
             dvar * 2 * (x - mean) / M + \
             dmean / M

        self.cache['dgamma'] = dgamma
        self.cache['dbeta'] = dbeta

        return dx, dgamma, dbeta

    def get_parameters(self):
        """
        Get trainable parameters

        Returns:
            dict: Dictionary containing gamma and beta parameters
        """
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

    def set_parameters(self, params):
        """
        Set trainable parameters

        Args:
            params (dict): Dictionary containing gamma and beta parameters
        """
        self.gamma = params['gamma']
        self.beta = params['beta']