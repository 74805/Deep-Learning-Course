import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt


# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x


# Define the fully-connected neural network
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# Define the convolutional neural network
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Define the custom classifier
class CustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define Fixed MobileNetV2 as feature extractor
class FixedMobileNetV2(nn.Module):
    def __init__(self):
        super(FixedMobileNetV2, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        for param in mobilenet.parameters():
            param.requires_grad = False
        self.features = mobilenet.features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


# Define Learned MobileNetV2 as feature extractor
class LearnedMobileNetV2(nn.Module):
    def __init__(self):
        super(LearnedMobileNetV2, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


# Function to count learnable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to train the model
def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy


# Function to evaluate the model
def evaluate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy


# Main function for training and evaluation
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomRotation(
                degrees=10
            ),  # Randomly rotate the image by up to 10 degrees
            transforms.ToTensor(),
        ]
    )

    # Validation/test data transforms (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    # Load STL-10 dataset with augmentation
    train_dataset = STL10(
        root="./data", split="train", transform=train_transform, download=True
    )
    val_dataset = STL10(
        root="./data", split="test", transform=val_transform, download=True
    )

    num_epochs = 10

    print(
        "Choose whether to train on different configuration or train once on a chosen architecture:"
    )
    print("1. Train on different configurations")
    print("2. Train once on a chosen architecture")
    choice = int(input("Enter your choice (1/2): "))

    if choice == 1:
        # Hyperparameters
        layer_sizes = [32, 64, 128, 256, 512]
        batch_sizes = [32, 64, 128, 256]
        optimizers = [optim.Adam, optim.SGD]
        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        regularization_coeffs = [0.1, 0.01, 0.001, 0.0001]
        architectures = [
            LogisticRegression,
            FullyConnectedNN,
            CNN,
            FixedMobileNetV2,
            LearnedMobileNetV2,
        ]

        # Sample random different configurations
        num_configs = 50
        sampled_configs = []
        for _ in range(num_configs):
            layer_size = layer_sizes[torch.randint(0, len(layer_sizes), (1,))]
            batch_size = batch_sizes[torch.randint(0, len(batch_sizes), (1,))]
            optimizer = optimizers[torch.randint(0, len(optimizers), (1,))]
            learning_rate = learning_rates[torch.randint(0, len(learning_rates), (1,))]
            regularization_coeff = regularization_coeffs[
                torch.randint(0, len(regularization_coeffs), (1,))
            ]
            architecture = architectures[torch.randint(0, len(architectures), (1,))]
            config = (
                layer_size,
                batch_size,
                optimizer,
                learning_rate,
                regularization_coeff,
                architecture,
            )

            while config in sampled_configs:
                layer_size = layer_sizes[torch.randint(0, len(layer_sizes), (1,))]
                batch_size = batch_sizes[torch.randint(0, len(batch_sizes), (1,))]
                optimizer = optimizers[torch.randint(0, len(optimizers), (1,))]
                learning_rate = learning_rates[
                    torch.randint(0, len(learning_rates), (1,))
                ]
                regularization_coeff = regularization_coeffs[
                    torch.randint(0, len(regularization_coeffs), (1,))
                ]
                architecture = architectures[torch.randint(0, len(architectures), (1,))]
                config = (
                    layer_size,
                    batch_size,
                    optimizer,
                    learning_rate,
                    regularization_coeff,
                    architecture,
                )
            sampled_configs.append(config)

        # Train on different configurations
        best_model = None
        for idx, config in enumerate(sampled_configs):
            print(f"\nConfiguration {idx+1}/{num_configs}")
            (
                layer_size,
                batch_size,
                optimizer,
                learning_rate,
                regularization_coeff,
                architecture,
            ) = config
            print(f"Layer Size: {layer_size}")
            print(f"Optimizer: {optimizer.__name__}")
            print(f"Learning Rate: {learning_rate}")
            print(f"Regularization Coefficient: {regularization_coeff}")
            print(f"Architecture: {architecture.__name__}")

            # Define data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            if architecture == LogisticRegression:
                input_size = 3 * 64 * 64
                num_classes = 10
                model = architecture(input_size, num_classes).to(device)
            elif architecture == FullyConnectedNN:
                input_size = 3 * 64 * 64
                num_classes = 10
                model = architecture(input_size, layer_size, num_classes).to(device)
            elif architecture == CNN:
                num_classes = 10
                model = architecture(num_classes).to(device)
            elif architecture == FixedMobileNetV2:
                num_classes = 10
                mobilenet = models.mobilenet_v2(pretrained=True)
                for param in mobilenet.parameters():
                    param.requires_grad = False
                mobilenet.classifier = CustomClassifier(num_classes)
                model = mobilenet.to(device)
            elif architecture == LearnedMobileNetV2:
                num_classes = 10
                mobilenet = models.mobilenet_v2(pretrained=True)
                mobilenet.classifier = CustomClassifier(num_classes)
                model = mobilenet.to(device)

            # Model, criterion, and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optimizer(model.parameters(), lr=learning_rate)

            # Training loop
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            for epoch in range(num_epochs):
                # Train the model
                train_loss, train_accuracy = train(
                    model, criterion, optimizer, train_loader, device
                )
                val_loss, val_accuracy = evaluate(model, criterion, val_loader, device)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )

            if best_model is None or val_losses[-1] < best_model[2][-1]:
                best_model = (
                    model,
                    train_losses,
                    val_losses,
                    train_accuracies,
                    val_accuracies,
                )

        # Plotting
        plt.figure(figsize=(num_epochs, 5))
        plt.plot(range(1, num_epochs + 1), best_model[1], label="Train Loss")
        plt.plot(range(1, num_epochs + 1), best_model[2], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(num_epochs, 5))
        plt.plot(range(1, num_epochs + 1), best_model[3], label="Train Accuracy")
        plt.plot(range(1, num_epochs + 1), best_model[4], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

    elif choice == 2:

        # Hyperparameters
        layer_size = 256
        batch_size = 64
        optimizer = optim.Adam
        learning_rate = 0.001
        regularization_coeff = 0.01

        # Define data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Choose the network architecture
        print("Choose the network architecture:")
        print("1. Logistic Regression")
        print("2. Fully-connected Neural Network")
        print("3. Convolutional Neural Network")
        print("4. MobileNetV2 as feature extractor (fixed) + Custom Classifier")
        print("5. MobileNetV2 as feature extractor (learned) + Custom Classifier")
        choice = int(input("Enter your choice (1/2/3/4/5): "))

        if choice == 1:
            input_size = 3 * 64 * 64
            num_classes = 10
            model = LogisticRegression(input_size, num_classes).to(device)
        elif choice == 2:
            input_size = 3 * 64 * 64
            hidden_size = 256
            num_classes = 10
            model = FullyConnectedNN(input_size, hidden_size, num_classes).to(device)
        elif choice == 3:
            num_classes = 10
            model = CNN(num_classes).to(device)
        elif choice == 4:
            num_classes = 10
            mobilenet = models.mobilenet_v2(pretrained=True)
            for param in mobilenet.parameters():
                param.requires_grad = False
            mobilenet.classifier = CustomClassifier(num_classes)
            model = mobilenet.to(device)
        elif choice == 5:
            num_classes = 10
            mobilenet = models.mobilenet_v2(pretrained=True)
            mobilenet.classifier = CustomClassifier(num_classes)
            model = mobilenet.to(device)
        else:
            print("Invalid choice. Exiting...")
            return

        # Print number of learnable parameters
        print(f"Number of learnable parameters: {count_parameters(model)}")

        # Model, criterion, and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(num_epochs):
            # Train the model
            train_loss, train_accuracy = train(
                model, criterion, optimizer, train_loader, device
            )
            val_loss, val_accuracy = evaluate(model, criterion, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        # Plotting
        plt.figure(figsize=(num_epochs, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
        plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(num_epochs, 5))
        plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
        plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()

    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    main()
