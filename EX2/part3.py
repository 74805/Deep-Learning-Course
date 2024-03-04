import threading
import numpy as np
import matplotlib.pyplot as plt


# Data Preprocessing
def normalize_data(train_data, test_data):
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data_normalized = (train_data - mean) / std
    test_data_normalized = (test_data - mean) / std
    return train_data_normalized, test_data_normalized


def train_val_split(data, labels, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    train_data, val_data = data[:split_index], data[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]
    return train_data, val_data, train_labels, val_labels


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        activation,
        output_size,
        learning_rate=0.01,
        reg_strength=0.01,
        dropout_rate=0,
    ):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_str = activation
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def activation(self, x):
        if self.activation_str == "relu":
            return np.maximum(0, x)
        elif self.activation_str == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_str == "tanh":
            return np.tanh(x)

    def forward(self, X):
        # Forward pass
        z1 = np.dot(X, self.W1) + self.b1
        h = self.activation(z1)

        # Dropout
        dropout_mask = None
        if self.dropout_rate > 0:
            dropout_mask = np.random.rand(*h.shape) < self.dropout_rate
            h *= dropout_mask / self.dropout_rate

        z2 = np.dot(h, self.W2) + self.b2
        probs = self.softmax(z2)
        return h, dropout_mask, probs

    def compute_loss(self, X, y, probs):
        num_examples = len(X)

        # Cross-entropy loss
        correct_logprobs = -np.log(probs[range(num_examples), y.astype(int)])
        data_loss = np.sum(correct_logprobs) / num_examples

        # L2 regularization term
        reg_loss = 0.5 * self.reg_strength * (np.sum(self.W1**2) + np.sum(self.W2**2))

        # Total loss
        loss = data_loss + reg_loss
        return loss

    def backward(self, X, y):
        num_examples = len(X)

        # Backpropagation
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        delta3 /= num_examples

        dW2 = np.dot(self.h.T, delta3)
        db2 = np.mean(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self.W2.T)
        if self.dropout_rate > 0:
            delta2 *= self.dropout_mask
        delta2[self.h <= 0] = 0

        dW1 = np.dot(X.T, delta2)
        db1 = np.mean(delta2, axis=0, keepdims=True)

        # Add regularization terms
        dW2 += self.reg_strength * self.W2
        dW1 += self.reg_strength * self.W1

        # Update gradients
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1.squeeze()
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2.squeeze()

    def train(self, X_train, y_train, X_val, y_val, num_epochs, batch_size):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Compute probabilities
        _, _, train_probs = self.forward(X_train)
        _, _, val_probs = self.forward(X_val)

        # Compute loss
        train_loss = self.compute_loss(X_train, y_train, train_probs)
        train_losses.append(train_loss)
        val_loss = self.compute_loss(X_val, y_val, val_probs)
        val_losses.append(val_loss)

        # Compute accuracy
        train_predictions = self.predict(X_train, train_probs)
        val_predictions = self.predict(X_val, val_probs)
        train_accuracy = accuracy(train_predictions, y_train)
        val_accuracy = accuracy(val_predictions, y_val)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"Before trainig: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}"
        )

        for epoch in range(num_epochs):
            # Mini-batch Gradient Descent
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Forward pass
                self.h, self.dropout_mask, self.probs = self.forward(X_batch)

                # Backpropagation
                self.backward(X_batch, y_batch)

            # Compute probabilities
            _, _, train_probs = self.forward(X_train)
            _, _, val_probs = self.forward(X_val)

            # Compute loss
            train_loss = self.compute_loss(X_train, y_train, train_probs)
            train_losses.append(train_loss)
            val_loss = self.compute_loss(X_val, y_val, val_probs)
            val_losses.append(val_loss)

            # Compute accuracy
            train_predictions = self.predict(X_train, train_probs)
            val_predictions = self.predict(X_val, val_probs)
            train_accuracy = accuracy(train_predictions, y_train)
            val_accuracy = accuracy(val_predictions, y_val)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}"
                )

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X, probs=None):
        if probs is None:
            _, _, probs = self.forward(X)
        return np.argmax(probs, axis=1)


def accuracy(predictions, labels):
    return np.mean(predictions == labels)


# Plot the loss curve and accuracy curve in the same figure
def plot_loss_curve(train_losses, val_losses, train_accuracies, val_accuracies, title):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(train_losses, label="Train")
    axs[0].plot(val_losses, label="Validation")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(train_accuracies, label="Train")
    axs[1].plot(val_accuracies, label="Validation")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    fig.suptitle(title)


def train_and_plot(
    X_train, y_train, X_val, y_val, input_size, output_size, configuration
):
    batch_size, learning_rate, reg_strength, hidden_size, activation, dropout_rate = (
        configuration
    )
    model = NeuralNetwork(
        input_size,
        hidden_size,
        activation,
        output_size,
        learning_rate=learning_rate,
        reg_strength=reg_strength,
        dropout_rate=dropout_rate,
    )
    train_losses, val_losses, train_accuracies, val_accuracies = model.train(
        X_train, y_train, X_val, y_val, num_epochs, batch_size
    )

    # Plot train and validation loss and accuracy curves as a function of the number of epochs
    plot_loss_curve(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        f"Loss and Accuracy curves with Batch size: {batch_size}, Learning rate: {learning_rate}, Regularization strength: {reg_strength}, hidden size: {hidden_size}, activation function: {activation}, Dropout rate: {dropout_rate:.4f}",
    )

    models.append(
        (
            model,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            configuration,
        )
    )

    print(
        f"Batch Size: {batch_size:.4f}, Learning Rate: {learning_rate:.4f}, Regularization Strength: {reg_strength:.4f}, hidden size: {hidden_size:.4f}, activation function: {activation}, Dropout rate: {dropout_rate:.4f}, Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.4f}"
    )


def find_best_model():
    best_model = None
    for model in models:
        if best_model is None or model[4][-1] > best_model[4][-1]:
            # Take the model with the highest validation accuracy
            best_model = model
    return best_model


# Load the data
train_data = np.genfromtxt("train.csv", delimiter=",", skip_header=1)
test_data = np.genfromtxt("test.csv", delimiter=",", skip_header=1)

X_train, X_val, y_train, y_val = train_val_split(train_data[:, 1:], train_data[:, 0])
X_train, X_val = normalize_data(X_train, X_val)
y_train = y_train.astype(int)

# Define configurations
input_size = X_train.shape[1]
hidden_sizes = [50, 100, 200]
activations = ["relu", "sigmoid", "tanh"]
output_size = 10
num_epochs = int(input("Enter the number of epochs: "))
batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.01, 0.1]
reg_strengths = [0.001, 0.01, 0.1]
dropout_rates = [0, 0.5]

# Create a figure to hold all the subplots
fig, axs = plt.subplots(
    len(batch_sizes), len(learning_rates) * len(reg_strengths), figsize=(15, 10)
)

# Initialize and train the neural network
number_of_tries = 30
configurations = set()
configuration = None
threads = []
models = []
for i in range(number_of_tries):
    print(f"Try {i + 1}/{number_of_tries}")

    # Generate a random configuration
    while configuration in configurations or configuration is None:
        hidden_size = np.random.choice(hidden_sizes)
        activation = np.random.choice(activations)
        batch_size = np.random.choice(batch_sizes)
        learning_rate = np.random.choice(learning_rates)
        reg_strength = np.random.choice(reg_strengths)
        dropout_rate = np.random.choice(dropout_rates)
        configuration = (
            batch_size,
            learning_rate,
            reg_strength,
            hidden_size,
            activation,
            dropout_rate,
        )

    configurations.add(configuration)

    train_and_plot(
        X_train, y_train, X_val, y_val, input_size, output_size, configuration
    )


# Find the best model
best_model = find_best_model()
model, train_losses, val_losses, train_accuracies, val_accuracies, configuration = (
    best_model
)
print(
    f"Best Model: Batch size: {configuration[0]}, Learning rate: {configuration[1]}, Regularization strength: {configuration[2]}, hidden size: {configuration[3]}, activation function: {configuration[4]}, Dropout rate: {configuration[5]:.4f}"
)

# Plot the best model's loss curve
plot_loss_curve(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    f"Best Model: Batch size: {configuration[0]}, Learning rate: {configuration[1]}, Regularization strength: {configuration[2]}, hidden size: {configuration[3]}, activation function: {configuration[4]}, Dropout rate: {configuration[5]:.4f}",
)

# Predict on test set
predictions = best_model[0].predict(test_data)

# Save predictions to a file
np.savetxt("NN_pred.csv", predictions, fmt="%d", delimiter=",")

# Plot train and validation loss curves as a function of the number of epochs
plt.show()
