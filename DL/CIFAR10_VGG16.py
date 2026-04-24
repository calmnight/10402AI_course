import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import multiprocessing as mp


# =========================
# Hyperparameters
# =========================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # On Windows, using 0 is the simplest and safest choice.
PIN_MEMORY = torch.cuda.is_available()


# =========================
# Data preprocessing
# ToTensor: [0, 255] --> [0, 1]
# Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)): standardization of [0, 1] with mean=0.5 and std=0.5
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# =========================
# Load CIFAR-10 dataset
# Training set: first 50000 images
# Test set: last 10000 images
# =========================
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# =========================
# Training function
# =========================
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# =========================
# Evaluation function
# =========================
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


# =========================
# Main training and plotting process
# =========================
def main():

    # # =========================
    # # Method 1: Learning from scratch (Initialize model weights randomly or by some theory)
    # # =========================
    # from torchvision import models
    # model = models.vgg16(weights=None)
    # model.classifier[6] = nn.Linear(4096, 10)
    # model = model.to(DEVICE)

    # =========================
    # Method 2: Transfer Learning (Use pre-trained model weights)
    # =========================
    from torchvision.models import vgg16, VGG16_Weights
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, 10)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"- Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_acc:.2f}%, "
                f"Test Accuracy: {test_acc:.2f}%")

    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss_accuracy_curve.png")


if __name__ == '__main__':
    mp.freeze_support()
    main()

