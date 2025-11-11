
"""
ECGR 4105 — HW5 — Problem 2
"""

import argparse, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="Housing.csv")
args = parser.parse_args()

# Load data 
df = pd.read_csv(args.csv)
cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
df = df[cols].dropna().reset_index(drop=True)

X = df[["area","bedrooms","bathrooms","stories","parking"]].values.astype("float32")
y = df["price"].values.astype("float32").reshape(-1,1)

# Simple 80/20 split 
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
cut = int(0.8*len(X))
tr, va = idx[:cut], idx[cut:]

Xtr_np, Xva_np = X[tr], X[va]
ytr_np, yva_np = y[tr], y[va]

# Standardize
Xmean = Xtr_np.mean(axis=0, keepdims=True)
Xstd  = Xtr_np.std(axis=0, keepdims=True) + 1e-8
ymean = ytr_np.mean(axis=0, keepdims=True)
ystd  = ytr_np.std(axis=0, keepdims=True) + 1e-8

Xtr = torch.tensor((Xtr_np - Xmean)/Xstd)
Xva = torch.tensor((Xva_np - Xmean)/Xstd)
ytr = torch.tensor((ytr_np - ymean)/ystd)
yva = torch.tensor((yva_np - ymean)/ystd)

# (a) Train tiny linear model: y = XW + b
W = torch.zeros((Xtr.shape[1], 1), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)

def mse(a,b): return ((a-b)**2).mean()

lr = 0.1
for epoch in range(1, 5001):
    yhat = Xtr @ W + b
    loss = mse(yhat, ytr)
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
    W.grad.zero_(); b.grad.zero_()
    if epoch % 500 == 0 or epoch == 1:
        print(f"[P2(a)] epoch {epoch:4d} train_loss={loss.item():.4f}")

# (b) Evaluate + one scatter plot in ORIGINAL $ units
with torch.no_grad():
    yva_pred_n = (Xva @ W + b).numpy().flatten()
yva_pred = yva_pred_n * ystd.flatten() + ymean.flatten()
yva_true = yva_np.flatten()

# metrics
mse_val = float(np.mean((yva_true - yva_pred)**2))
ss_res = float(np.sum((yva_true - yva_pred)**2))
ss_tot = float(np.sum((yva_true - yva_true.mean())**2))
r2 = 1 - ss_res/ss_tot

print(f"[P2(b)] Validation MSE: {mse_val:.2f}")
print(f"[P2(b)] Validation R^2: {r2:.3f}")

plt.figure()
plt.scatter(yva_true, yva_pred)
plt.xlabel("True Price"); plt.ylabel("Predicted Price")
plt.title("HW5 Problem 2 — Linear Regression (vanilla GD)")
plt.savefig("p2_pred_vs_true.png", bbox_inches="tight")
print("Saved plot: p2_pred_vs_true.png")
