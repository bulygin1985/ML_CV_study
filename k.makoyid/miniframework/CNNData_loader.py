from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import cifar10


class DataLoader:
    @staticmethod
    def load_cifar10_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-10 dataset, using only 10,000 images total.
        Split ratio: 70% training (7,000), 10% validation (1,000), 20% test (2,000)

        Returns:
            Tuple containing training, validation, and test data with their labels
        """
        # Load CIFAR-10 data
        (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

        # Select only 10,000 images (8,000 from training, 2,000 from test)
        total_train_samples = 37000
        total_test_samples = 8000

        # Randomly select indices to ensure diverse sampling
        train_indices = np.random.choice(len(x_train_full), total_train_samples, replace=False)
        test_indices = np.random.choice(len(x_test), total_test_samples, replace=False)

        x_train_full = x_train_full[train_indices]
        y_train_full = y_train_full[train_indices]
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]

        # Preprocess data - normalize to [0,1] and maintain 4D shape (N,C,H,W)
        x_train_full = x_train_full.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Transpose from (N,H,W,C) to (N,C,H,W) format
        x_train_full = np.transpose(x_train_full, (0, 3, 1, 2))
        x_test = np.transpose(x_test, (0, 3, 1, 2))

        # One-hot encode labels
        y_train_full = np.eye(10)[y_train_full.squeeze()]
        y_test = np.eye(10)[y_test.squeeze()]

        # Split training set into train and validation
        # From 8,000 training samples: 7,000 for training, 1,000 for validation
        num_training = 34000
        num_validation = 3000

        # Keep 4D shape for CNN input
        x_train = x_train_full[:num_training]
        y_train = y_train_full[:num_training].T

        x_val = x_train_full[num_training:num_training + num_validation]
        y_val = y_train_full[num_training:num_training + num_validation].T

        x_test = x_test
        y_test = y_test.T

        # Print dataset sizes for verification
        print(f"Training set size: {x_train.shape[0]} images")
        print(f"Validation set size: {x_val.shape[0]} images")
        print(f"Test set size: {x_test.shape[0]} images")
        print(f"Total dataset size: {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]} images")

        return x_train, y_train, x_val, y_val, x_test, y_test