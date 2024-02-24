import numpy as np
import matplotlib.pyplot as plt

# Load the data from train.csv
data = np.genfromtxt("train.csv", delimiter=",", dtype=None, encoding=None)

# Extract labels and features
labels = data[1:, 0]
features = data[1:, 1:].astype(np.float32)

# Sort the data by labels
indices = np.argsort(labels)
labels = labels[indices].astype(np.int32)
features = features[indices].astype(np.float32)

# Define class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Initialize a figure
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 30))

# plot an example of each class
for i, class_name in enumerate(class_names):
    # Select indices of examples belonging to the current class
    class_indices = np.where(labels == i)[0]

    # Plot the first example of the class
    ax = axes[i // 2, i % 2]
    ax.imshow(features[class_indices[0]].reshape(28, 28), cmap="gray")
    ax.axis("off")
    ax.set_title(class_name)

# Adjust layout
plt.subplots_adjust(hspace=100)
plt.tight_layout()
plt.show()
