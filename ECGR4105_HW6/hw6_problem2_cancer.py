# hw6_problem2_cancer.py

from sklearn.datasets import load_breast_cancer
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load built-in dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Network
class CancerNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

model = CancerNet(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
lr = 0.01
epochs = 50

train_losses = []
val_losses = []

# Manual GD
for ep in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
        model.zero_grad()

        total_loss += loss.item() * xb.size(0)

    train_losses.append(total_loss / len(train_loader.dataset))

    # Val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)

    val_losses.append(val_loss / len(val_loader.dataset))

    print(f"Epoch {ep+1}: Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}")

# Metrics
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_val_t)).numpy().ravel()
y_pred = (probs >= 0.5).astype(int)

print("NN Accuracy:", accuracy_score(y_val, y_pred))
print("NN Precision:", precision_score(y_val, y_pred))
print("NN Recall:", recall_score(y_val, y_pred))
print("NN F1:", f1_score(y_val, y_pred))

# Baselines
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
svm = SVC(kernel="rbf", C=1.0)
svm.fit(X_train, y_train)

print("\nLogistic Regression Accuracy:", accuracy_score(y_val, logreg.predict(X_val)))
print("SVM Accuracy:", accuracy_score(y_val, svm.predict(X_val)))

# Plot
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.title("Cancer NN Loss")
plt.legend()
plt.show()
