from Data_loader import DataLoader
from Network import NeuralNetwork
from Model_train import ModelTrainer
from Visualizer import Visualizer

if __name__ == "__main__":
    print("Starting the program...")
    # Завантаження та підготовка даних
    print("Loading data...")
    x_train, y_train, x_test, y_test = DataLoader.load_cifar10_data()
    x_train = x_train[:, :37000]  # Використання лише перших 37000 прикладів для навчання
    y_train = y_train[:, :37000]
    print("Data loaded successfully.")

    print("Initializing model and trainer...")
    # Гіперпараметри
    input_size = x_train.shape[0]
    hidden_size = 64
    num_classes = y_train.shape[0]
    initial_lr = 1e-3
    batch_size = 64
    lambda_wd = 0.01

    # Створення моделі та тренера
    model = NeuralNetwork(input_size, hidden_size, num_classes)
    trainer = ModelTrainer(model, x_train, y_train, x_test, y_test, initial_lr, batch_size, lambda_wd)
    print("Starting training...")
    # Навчання моделі
    train_losses, train_accuracies, test_losses, test_accuracies = trainer.train_model_with_lr_decay()
    print("Training completed.")
    # Візуалізація результатів
    Visualizer.plot_results(train_losses, train_accuracies, test_losses, test_accuracies)

    print(f"Final train accuracy: {train_accuracies[-1]}")
    print(f"Final test accuracy: {test_accuracies[-1]}")