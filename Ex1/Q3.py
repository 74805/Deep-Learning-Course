import numpy as np


def f(x, w, U, b1, b2):
    h = np.maximum(np.dot(U.T, x) + b1, 0)
    return np.dot(w.T, h) + b2


def create_data_set():
    data = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = [-1, 1, 1, -1]
    data_set = dict(zip(data, labels))
    return data_set


def squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


dataset = create_data_set()
