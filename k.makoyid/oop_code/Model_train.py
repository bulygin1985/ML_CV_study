import numpy as np


class ModelTrainer:
    def __init__(self, model, X_train, Y_train, X_test, Y_test, initial_lr, batch_size, lambda_wd):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.lambda_wd = lambda_wd

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        ce_loss = -1 / m * np.sum(Y * np.log(A2 + 1e-8))
        l2_regularization = self.lambda_wd * (np.sum(np.square(self.model.W1)) + np.sum(np.square(self.model.W2))) / (
                    2 * m)
        return ce_loss + l2_regularization

    def compute_accuracy(self, X, Y):
        A2, _ = self.model.forward_propagation(X)
        predictions = np.argmax(A2, axis=0)
        true_labels = np.argmax(Y, axis=0)
        return np.mean(predictions == true_labels)

    def train_model_with_lr_decay(self):
        m = self.X_train.shape[1]
        num_batches = m // self.batch_size

        lr_schedule = [
            (120, self.initial_lr),
            (50, (self.initial_lr / 100) * 4),
            (50, self.initial_lr / 1000),
            (50, self.initial_lr / 10000)
        ]

        total_epochs = sum(epochs for epochs, _ in lr_schedule)
        current_epoch = 0

        for epochs, lr in lr_schedule:
            for _ in range(epochs):
                for j in range(num_batches):
                    start = j * self.batch_size
                    end = start + self.batch_size
                    X_batch = self.X_train[:, start:end]
                    Y_batch = self.Y_train[:, start:end]

                    A2, cache = self.model.forward_propagation(X_batch)
                    gradients = self.model.backward_propagation(X_batch, Y_batch, cache, self.lambda_wd)
                    self.model.update_parameters(gradients, lr)

                current_epoch += 1
                if current_epoch % 10 == 0:
                    self.record_metrics()
                    self.print_progress(current_epoch, lr)

            if current_epoch % 50 == 0 and current_epoch < total_epochs:
                lr /= 10
                print(f"Reducing learning rate to {lr:.1e}")

        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies

    def record_metrics(self):
        train_A2, _ = self.model.forward_propagation(self.X_train)
        test_A2, _ = self.model.forward_propagation(self.X_test)

        train_loss = self.compute_loss(train_A2, self.Y_train)
        test_loss = self.compute_loss(test_A2, self.Y_test)

        train_accuracy = self.compute_accuracy(self.X_train, self.Y_train)
        test_accuracy = self.compute_accuracy(self.X_test, self.Y_test)

        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)

    def print_progress(self, epoch, lr):
        print(f"Epoch {epoch}, LR: {lr:.1e}, Train Loss: {self.train_losses[-1]:.4f}, "
              f"Train Accuracy: {self.train_accuracies[-1]:.4f}, Test Loss: {self.test_losses[-1]:.4f}, "
              f"Test Accuracy: {self.test_accuracies[-1]:.4f}")