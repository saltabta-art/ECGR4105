# hw7_cifar_cnn_resnet.py
# Ali - ECGR4105 HW7
# CNNs and ResNet on CIFAR-10

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# CIFAR-10 Dataloaders

def get_cifar10_loaders(batch_size=128):
    # Standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # 90% train, 10% val from training set
    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# Helper: count parameters

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Problem 1(a): Baseline CNN

class SimpleCNN1(nn.Module):
    """
    Problem 1(a): baseline CNN with 2 conv + pool blocks.
    3x32x32 -> (Conv/ReLU/Pool)x2 -> FC -> output(10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 3x32x32 -> 16x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16x16 -> 32x8x8
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Problem 1(b): Deeper CNN

class SimpleCNN2(nn.Module):
    """
    Problem 1(b): deeper CNN with 3 conv + pool blocks.
    Adds one extra conv + pool before FC.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pools: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 3x32x32 -> 16x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16x16 -> 32x8x8
        x = self.pool(F.relu(self.conv3(x)))   # 32x8x8   -> 64x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ResNet block and ResNet-10

class ResBlock(nn.Module):
    """
    Basic residual block:
    input -> Conv -> ReLU -> Conv -> +input -> ReLU
    Optional batchnorm and dropout.
    """
    def __init__(self, channels, use_bn=False, use_dropout=False, p_drop=0.3):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)
        if use_dropout:
            self.dropout = nn.Dropout2d(p_drop)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        out = out + identity
        out = F.relu(out)
        return out

class ResNet10(nn.Module):
    """
    Very simple ResNet-10 style model:
    Conv -> 10 ResBlocks -> Global AvgPool -> FC(16->10)
    Channels kept at 16 for simplicity.
    """
    def __init__(self, use_bn=False, use_dropout=False, p_drop=0.3):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        blocks = []
        for _ in range(10):
            blocks.append(ResBlock(16, use_bn=use_bn, use_dropout=use_dropout, p_drop=p_drop))
        self.blocks = nn.Sequential(*blocks)

        self.fc_out = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv_in(x))        # 3x32x32 -> 16x32x32
        x = self.blocks(x)                 # residual blocks
        x = F.adaptive_avg_pool2d(x, (1, 1))   # global avg pool -> 16x1x1
        x = x.view(x.size(0), -1)          # flatten to 16
        x = self.fc_out(x)                 # -> 10
        return x


# Training / Evaluation helper

def train_model(model, train_loader, val_loader,
                num_epochs=300, lr=0.001, weight_decay=0.0,
                description=""):
    """
    Manual gradient descent training loop.
    Returns: (train_losses, val_accuracies, training_time)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    print("\n=== Training", description, "===")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            # L2 regularization (weight decay) manually
            if weight_decay > 0:
                l2 = 0.0
                for p in model.parameters():
                    l2 = l2 + torch.sum(p * p)
                loss = loss + weight_decay * l2

            loss.backward()

            # manual gradient step
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
            model.zero_grad()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)

        # print occasionally so 300 epochs is not crazy
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"Finished training {description} in {total_time:.1f} seconds")
    print("Final Train Loss:", train_losses[-1])
    print("Final Val Accuracy:", val_accs[-1])

    return train_losses, val_accs, total_time


# Main: run all experiments

if __name__ == "__main__":
    batch_size = 128
    lr = 0.001
    epochs = 300

    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    # -------- Problem 1(a): SimpleCNN1 --------
    cnn1 = SimpleCNN1()
    print("SimpleCNN1 params:", count_params(cnn1))
    train_model(cnn1, train_loader, val_loader,
                num_epochs=epochs, lr=lr, weight_decay=0.0,
                description="Problem 1(a) - SimpleCNN1")

    # -------- Problem 1(b): SimpleCNN2 (deeper) --------
    cnn2 = SimpleCNN2()
    print("SimpleCNN2 params:", count_params(cnn2))
    train_model(cnn2, train_loader, val_loader,
                num_epochs=epochs, lr=lr, weight_decay=0.0,
                description="Problem 1(b) - SimpleCNN2 (deeper)")

    # -------- Problem 2(a): ResNet-10 baseline (no reg) --------
    resnet_base = ResNet10(use_bn=False, use_dropout=False)
    print("ResNet10 baseline params:", count_params(resnet_base))
    train_model(resnet_base, train_loader, val_loader,
                num_epochs=epochs, lr=lr, weight_decay=0.0,
                description="Problem 2(a) - ResNet10 baseline")

    # -------- Problem 2(b): ResNet-10 + Weight Decay --------
    resnet_wd = ResNet10(use_bn=False, use_dropout=False)
    print("ResNet10 + weight decay params:", count_params(resnet_wd))
    train_model(resnet_wd, train_loader, val_loader,
                num_epochs=epochs, lr=lr, weight_decay=0.001,
                description="Problem 2(b) - ResNet10 + weight decay (lambda=0.001)")

    # -------- Problem 2(b): ResNet-10 + Dropout --------
    resnet_do = ResNet10(use_bn=False, use_dropout=True, p_drop=0.3)
    print("ResNet10 + dropout params:", count_params(resnet_do))
    train_model(resnet_do, train_loader, val_loader,
                num_epochs=epochs, lr=lr, weight_decay=0.0,
                description="Problem 2(b) - ResNet10 + dropout (p=0.3)")

    # -------- Problem 2(b): ResNet-10 + BatchNorm --------
    resnet_bn = ResNet10(use_bn=True, use_dropout=False)
    print("ResNet10 + batchnorm params:", count_params(resnet_bn))
    train_model(resnet_bn, train_loader, val_loader,
                num_epochs=epochs, lr=lr, weight_decay=0.0,
                description="Problem 2(b) - ResNet10 + batch normalization")
