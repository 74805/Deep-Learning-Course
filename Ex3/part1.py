import torch
import torchvision.transforms as transforms
from torchvision.datasets import STL10
import matplotlib.pyplot as plt


def visualize_data(dataset, class_names):
    fig, axes = plt.subplots(nrows=len(class_names), ncols=4, figsize=(10, 30))

    for i, class_name in enumerate(class_names):
        indices = [idx for idx, label in enumerate(dataset.labels) if label == i]

        # Plot 4 examples
        for j, idx in enumerate(indices[:4]):
            img, _ = dataset[idx]
            axes[i, j].imshow(img.permute(1, 2, 0))
            axes[i, j].set_title(class_name)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


class_names = [
    "Airplane",
    "Bird",
    "Car",
    "Cat",
    "Deer",
    "Dog",
    "Horse",
    "Monkey",
    "Ship",
    "Truck",
]

# Load the STL-10 dataset
transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ]
)
train_dataset = STL10(root="./data", split="train", transform=transform, download=True)
test_dataset = STL10(root="./data", split="test", transform=transform, download=True)

visualize_data(train_dataset, class_names)
