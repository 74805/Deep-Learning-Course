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


# Model Implementation
class LogisticRegression:
    def __init__(self, input_size, num_classes, learning_rate=0.01, reg_strength=0.01):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.weights = np.zeros((input_size, num_classes))
        self.bias = np.zeros(num_classes)

    def softmax(self, logits):
        # Subtract the maximum value of logits
        max_logit = np.max(logits)
        shifted_logits = logits - max_logit  # to avoid overflow
        exp_shifted_logits = np.exp(shifted_logits)
        softmax_probs = exp_shifted_logits / np.sum(
            exp_shifted_logits, axis=1, keepdims=True
        )
        return softmax_probs

    def L2(self, X, y):
        logits = np.dot(X, self.weights) + self.bias
        probs = self.softmax(logits)

        # Compute cross-entropy loss
        y = y.astype(int)
        cross_entropy_loss = -np.sum(np.log(probs[range(len(y)), y])) / len(y)

        # Compute L2 regularization term
        l2_regularization_term = 0.5 * self.reg_strength * np.sum(self.weights**2)

        # Return total loss (cross-entropy loss + L2 regularization term)
        return cross_entropy_loss + l2_regularization_term

    def compute_gradients(self, X, y, probs):
        # Compute one-hot encoded labels
        y = y.astype(int)
        y_onehot = np.zeros(probs.shape)
        y_onehot[range(len(y)), y] = 1

        # Compute gradient of the loss with respect to the weights
        grad = -np.dot(X.T, y_onehot - probs) / len(X)

        # Add L2 regularization term to the gradient
        grad += self.reg_strength * self.weights

        return grad

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def train(self, X, y):
        probs = self.predict(X)
        grad = self.compute_gradients(X, y, probs)
        self.weights -= self.learning_rate * grad

        # Reshape y to match the shape of probs
        y_reshaped = y[:, np.newaxis]

        # Subtract y from probs element-wise and update bias
        self.bias -= self.learning_rate * np.mean(probs - y_reshaped, axis=0)


# Training
def train_model(model, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=64):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    num_batches = len(X_train) // batch_size
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X_batch, y_batch = X_train[start:end], y_train[start:end]
            model.train(X_batch, y_batch)

        model.train(X_train, y_train)
        train_loss = model.L2(X_train, y_train)
        val_loss = model.L2(X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        train_accuracy = accuracy(train_predictions, y_train)
        val_accuracy = accuracy(val_predictions, y_val)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

    return train_losses, val_losses, train_accuracies, val_accuracies


def mini_batch_gradient_descent(self, X, y, batch_size=64, num_epochs=100):
    num_batches = len(X) // batch_size
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X_batch, y_batch = X[start:end], y[start:end]
            self.train(X_batch, y_batch)


def accuracy(predictions, labels):
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(predicted_labels == labels)


# Load the data
train_data = np.genfromtxt("train.csv", delimiter=",", skip_header=1)
test_data = np.genfromtxt("test.csv", delimiter=",", skip_header=1)

X_train, X_val, y_train, y_val = train_val_split(train_data[:, 1:], train_data[:, 0])
X_train, X_val = normalize_data(X_train, X_val)

input_size = X_train.shape[1]
num_classes = 10

# Initialize and train the model
model = LogisticRegression(input_size, num_classes)
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, X_train, y_train, X_val, y_val
)

for i in range(len(train_losses)):
    print(
        f"Epoch {i + 1}, Train Loss: {train_losses[i]:.4f}, Train Acc: {train_accuracies[i]:.4f}, Val Loss: {val_losses[i]:.4f}, Val Acc: {val_accuracies[i]:.4f}"
    )

# Plot train and validation loss and accuracy curves as a function of the number of epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Generate predictions for test set
test_predictions = model.predict(test_data)
predicted_labels = np.argmax(test_predictions, axis=1)

# Save predictions to a file
np.savetxt("lr_pred.csv", predicted_labels, fmt="%d", delimiter="\n")
