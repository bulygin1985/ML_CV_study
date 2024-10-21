#Visualizer.py
import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def plot_results(train_losses, train_accuracies, test_losses, test_accuracies):
        plt.figure(figsize=(12, 8))

        plt.subplot(221)
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.xlabel('Iterations (x10)')
        plt.ylabel('Loss')
        plt.title('Loss vs Iterations')
        plt.legend()

        plt.subplot(222)
        plt.plot(train_accuracies, label='Train')
        plt.plot(test_accuracies, label='Test')
        plt.xlabel('Iterations (x10)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.legend()

        plt.tight_layout()
        plt.show()