
"""
ECGR 4105 — HW5 — Problem 3
"""

import argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="Housing.csv")
args = parser.parse_args()

# Load + split + standardize
df = pd.read_csv(args.csv)
cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
df = df[cols].dropna().reset_index(drop=True)

X = df[["area","bedrooms","bathrooms","stories","parking"]].values.astype("float32")
y = df["price"].values.astype("float32").reshape(-1,1)

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
cut = int(0.8*len(X))
tr, va = idx[:cut], idx[cut:]

Xtr_np, Xva_np = X[tr], X[va]
ytr_np, yva_np = y[tr], y[va]

Xmean = Xtr_np.mean(axis=0, keepdims=True)
Xstd  = Xtr_np.std(axis=0, keepdims=True) + 1e-8
ymean = ytr_np.mean(axis=0, keepdims=True)
ystd  = ytr_np.std(axis=0, keepdims=True) + 1e-8

Xtr = torch.tensor((Xtr_np - Xmean)/Xstd)
Xva = torch.tensor((Xva_np - Xmean)/Xstd)
ytr = torch.tensor((ytr_np - ymean)/ystd)
yva = torch.tensor((yva_np - ymean)/ystd)

def mse(a,b): return ((a-b)**2).mean()

# Helper: manual gradient descent on a small network
def train_vanilla_gd(model, Xtr, ytr, Xva, yva, epochs=200, lr=0.01, tag=""):
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(1, epochs+1):
        yhat = model(Xtr)
        loss = mse(yhat, ytr)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
        for p in params:
            p.grad.zero_()
        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                vloss = mse(model(Xva), yva).item()
            print(f"[{tag}] epoch {epoch:3d} train_loss={loss.item():.4f} val_loss={vloss:.4f}")
    # predictions in original units
    with torch.no_grad():
        yva_pred_n = model(Xva).numpy().flatten()
    yva_pred = yva_pred_n * ystd.flatten() + ymean.flatten()
    yva_true = yva_np.flatten()
    return yva_true, yva_pred

# (a) one hidden layer
in_dim = Xtr.shape[1]
model_3a = nn.Sequential(
    nn.Linear(in_dim, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
y_true_3a, y_pred_3a = train_vanilla_gd(model_3a, Xtr, ytr, Xva, yva, epochs=200, lr=0.01, tag="P3(a)")

# metrics and plot
mse_3a = float(np.mean((y_true_3a - y_pred_3a)**2))
ss_res = float(np.sum((y_true_3a - y_pred_3a)**2))
ss_tot = float(np.sum((y_true_3a - np.mean(y_true_3a))**2))
r2_3a = 1 - ss_res/ss_tot
print(f"[P3(a)] Validation MSE: {mse_3a:.2f}  R^2: {r2_3a:.3f}")

plt.figure()
plt.scatter(y_true_3a, y_pred_3a)
plt.xlabel("True Price"); plt.ylabel("Predicted Price")
plt.title("HW5 Problem 3(a): 1-hidden-layer NN (vanilla GD)")
plt.savefig("p3a_pred_vs_true.png", bbox_inches="tight")
print("Saved plot: p3a_pred_vs_true.png")

# (b) three hidden layers 
model_3b = nn.Sequential(
    nn.Linear(in_dim, 8),
    nn.ReLU(),
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
y_true_3b, y_pred_3b = train_vanilla_gd(model_3b, Xtr, ytr, Xva, yva, epochs=200, lr=0.01, tag="P3(b)")

mse_3b = float(np.mean((y_true_3b - y_pred_3b)**2))
ss_res = float(np.sum((y_true_3b - y_pred_3b)**2))
ss_tot = float(np.sum((y_true_3b - np.mean(y_true_3b))**2))
r2_3b = 1 - ss_res/ss_tot
print(f"[P3(b)] Validation MSE: {mse_3b:.2f}  R^2: {r2_3b:.3f}")

plt.figure()
plt.scatter(y_true_3b, y_pred_3b)
plt.xlabel("True Price"); plt.ylabel("Predicted Price")
plt.title("HW5 Problem 3(b): 3-hidden-layer NN (vanilla GD)")
plt.savefig("p3b_pred_vs_true.png", bbox_inches="tight")
print("Saved plot: p3b_pred_vs_true.png")
