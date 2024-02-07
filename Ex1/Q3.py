import numpy as np
import matplotlib.pyplot as plt


def f(x, w, U, b1, b2):
    h = np.maximum(np.dot(U.T, x) + b1, 0)
    return np.dot(w.T, h) + b2, h


def create_data_set():
    data = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = [-1, 1, 1, -1]
    data_set = dict(zip(data, labels))
    return data_set


def squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient(x, y_true, w, U, b1, b2):
    y_pred, h = f(x, w, U, b1, b2)
    loss = squared_loss(y_true, y_pred)

    dw = -2 * h * (y_true - y_pred)
    db2 = -2 * (y_true - y_pred)
    dU = -2 * np.outer(x, w * (y_true - y_pred) * (h > 0))
    db1 = -2 * w * (y_true - y_pred) * (h > 0)

    return dw, dU, db1, db2, loss


def gradient_descent(dataset, learning_rate, epochs):
    # Initialize weights
    w = np.random.randn(2)
    U = np.random.randn(2, 2)
    b1 = np.random.randn(2)
    b2 = np.random.randn(1)

    loss_values = []

    # Train the model
    for epoch in range(epochs):
        for x, y_true in dataset.items():
            dw, dU, db1, db2, loss = gradient(x, y_true, w, U, b1, b2)

            w -= learning_rate * dw
            U -= learning_rate * dU
            b1 -= learning_rate * db1
            b2 -= learning_rate * db2

            loss_values.append(loss)

    return w, U, b1, b2, loss_values


dataset = create_data_set()
w, U, b1, b2, loss_values = gradient_descent(dataset, 0.001, 10000)
print(f"w: {w}\nU: {U}\nb1: {b1}\nb2: {b2}")

# Test the model
print(f"f(0, 0) = {f(np.array([0, 0]), w, U, b1, b2)[0]}")
print(f"f(0, 1) = {f(np.array([0, 1]), w, U, b1, b2)[0]}")
print(f"f(1, 0) = {f(np.array([1, 0]), w, U, b1, b2)[0]}")
print(f"f(1, 1) = {f(np.array([1, 1]), w, U, b1, b2)[0]}")

# Plot the loss value as a function of the number of epochs
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss as a function of the number of epochs")
plt.show()
