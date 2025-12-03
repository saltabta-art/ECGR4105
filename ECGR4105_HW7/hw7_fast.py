# hw7_fast.py
# Faster HW7: trains all models ~15 epochs on 20% CIFAR subset

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)


# Load CIFAR-10 (small subset)

def get_small_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Use only 20% of train set = 10,000 samples
    small_size = int(0.20 * len(full))
    subset_indices = list(range(small_size))
    small = Subset(full, subset_indices)

    # Split into train/val (80/20)
    train_size = int(0.8 * small_size)
    val_size = small_size - train_size
    train_set, val_set = random_split(small, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = get_small_loaders()


# Helper

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_loader, val_loader, epochs=15, lr=0.001, wd=0.0, name=""):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    losses = []
    accs = []

    start = time.time()
    print(f"\n=== Training {name} ===")

    for ep in range(epochs):
        model.train()
        total = 0
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            loss = crit(out, labels)

            if wd > 0:
                L2 = sum((p**2).sum() for p in model.parameters())
                loss = loss + wd * L2

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
            model.zero_grad()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        avg_loss = total_loss / total
        losses.append(avg_loss)

        # Validation accuracy
        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total_val += labels.size(0)
        acc = correct / total_val
        accs.append(acc)

        print(f"Epoch {ep+1}/{epochs} - Loss {avg_loss:.4f} - Val Acc {acc:.4f}")

    end = time.time()
    print(f"Total time: {end-start:.2f}s")
    print("Final Loss:", losses[-1])
    print("Final Val Acc:", accs[-1])

    return losses, accs, end-start


# Models

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3,16,3,padding=1)
        self.c2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*8*8,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,16,3,padding=1)
        self.c2=nn.Conv2d(16,32,3,padding=1)
        self.c3=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*4*4,128)
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=self.pool(F.relu(self.c1(x)))
        x=self.pool(F.relu(self.c2(x)))
        x=self.pool(F.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class ResBlock(nn.Module):
    def __init__(self, ch, bn=False, dropout=False):
        super().__init__()
        self.bn = bn
        self.do = dropout
        self.c1 = nn.Conv2d(ch,ch,3,padding=1)
        self.c2 = nn.Conv2d(ch,ch,3,padding=1)
        if bn:
            self.b1 = nn.BatchNorm2d(ch)
            self.b2 = nn.BatchNorm2d(ch)
        if dropout:
            self.drop = nn.Dropout2d(0.3)

    def forward(self,x):
        id = x
        out = self.c1(x)
        if self.bn: out = self.b1(out)
        out = F.relu(out)
        if self.do: out = self.drop(out)
        out = self.c2(out)
        if self.bn: out = self.b2(out)
        out = out + id
        return F.relu(out)

class ResNet10(nn.Module):
    def __init__(self, bn=False, dropout=False):
        super().__init__()
        self.inp = nn.Conv2d(3,16,3,padding=1)
        self.blocks = nn.Sequential(*[
            ResBlock(16,bn=bn,dropout=dropout) for _ in range(10)
        ])
        self.fc = nn.Linear(16,10)

    def forward(self,x):
        x = F.relu(self.inp(x))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = x.view(x.size(0),-1)
        return self.fc(x)


# Run all experiments

print("\n=== Running HW7 FAST version ===")

# Problem 1(a)
m1 = CNN1()
print("CNN1 params:", count_params(m1))
train(m1, train_loader, val_loader, name="CNN1")

# Problem 1(b)
m2 = CNN2()
print("CNN2 params:", count_params(m2))
train(m2, train_loader, val_loader, name="CNN2 deeper")

# Problem 2(a) ResNet-10
res_base = ResNet10()
print("ResNet10 params:", count_params(res_base))
train(res_base, train_loader, val_loader, name="ResNet10 baseline")

# Problem 2(b) Weight decay
res_wd = ResNet10()
print("ResNet10 weight decay params:", count_params(res_wd))
train(res_wd, train_loader, val_loader, wd=0.001, name="ResNet10 + weight decay")

# Problem 2(b) Dropout
res_do = ResNet10(dropout=True)
print("ResNet10 dropout params:", count_params(res_do))
train(res_do, train_loader, val_loader, name="ResNet10 + dropout")

# Problem 2(b) BatchNorm
res_bn = ResNet10(bn=True)
print("ResNet10 BN params:", count_params(res_bn))
train(res_bn, train_loader, val_loader, name="ResNet10 + batchnorm")
