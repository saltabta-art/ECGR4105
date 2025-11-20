# hw6_problem3_cifar10.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset_full = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# make a small validation set from train set
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# Helper: flatten images
def flatten_batch(x):
    # x: [batch, 3, 32, 32]
    return x.view(x.size(0), -1)  # [batch, 3072]

input_dim = 3 * 32 * 32
num_classes = 10

# Model (a): 1 hidden layer (512)
class FCN_1Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = flatten_batch(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Model (b): 3 hidden layers (512-256-128) 
class FCN_3Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = flatten_batch(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

# Training and evaluation functions
def train_one_epoch(model, loader, criterion, lr):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        model.zero_grad()

        total_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

# Part (a): 1-hidden-layer model
def run_part_a():
    model = FCN_1Hidden()
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    num_epochs = 30  

    train_losses = []
    val_accuracies = []

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, lr)
        val_acc = evaluate_accuracy(model, val_loader)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        print(f"[Part A] Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    total_time = time.time() - start_time
    print(f"[Part A] Total training time: {total_time:.2f} seconds")

    return model, train_losses, val_accuracies, total_time

# Part (b): 3-hidden-layer model (300 epochs)
def run_part_b():
    model = FCN_3Hidden()
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    num_epochs = 300  

    train_losses = []
    val_accuracies = []

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, lr)
        val_acc = evaluate_accuracy(model, val_loader)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"[Part B] Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    total_time = time.time() - start_time
    print(f"[Part B] Total training time: {total_time:.2f} seconds")

    return model, train_losses, val_accuracies, total_time

if __name__ == "__main__":
    # run part a
    model_a, train_losses_a, val_accs_a, time_a = run_part_a()
    # run part b
    model_b, train_losses_b, val_accs_b, time_b = run_part_b()
