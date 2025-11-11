
"""
ECGR 4105 — HW5 — Problem 1
"""

import torch
import matplotlib.pyplot as plt

# Data
t_c = torch.tensor([0.5,14.0,15.0,28.0,11.0,8.0,3.0,-4.0,6.0,13.0,21.0], dtype=torch.float32)
t_u = torch.tensor([35.7,55.9,58.2,81.9,56.3,48.9,33.9,21.8,48.4,60.4,68.4], dtype=torch.float32)

# simple normalization for stability
x = t_u / t_u.mean()
y = t_c

def mse(a,b):
    return ((a-b)**2).mean()

# (a) Linear model: y = w*x + b 
w_lin = torch.tensor(0.1, requires_grad=True)
b_lin = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for epoch in range(1, 5001):  # 5000 steps
    yhat = w_lin * x + b_lin
    loss = mse(yhat, y)
    loss.backward()
    with torch.no_grad():
        w_lin -= lr * w_lin.grad
        b_lin -= lr * b_lin.grad
    w_lin.grad.zero_(); b_lin.grad.zero_()
    if epoch % 500 == 0 or epoch == 1:
        print(f"[P1(a)] epoch {epoch:4d} loss={loss.item():.4f} w={w_lin.item():.4f} b={b_lin.item():.4f}")

# (b) Quadratic: y = w2*x^2 + w1*x + b 
w2 = torch.tensor(0.0, requires_grad=True)
w1 = torch.tensor(0.1, requires_grad=True)
bq = torch.tensor(0.0, requires_grad=True)

for epoch in range(1, 5001):
    yhat = w2*(x**2) + w1*x + bq
    loss = mse(yhat, y)
    loss.backward()
    with torch.no_grad():
        w2 -= lr * w2.grad
        w1 -= lr * w1.grad
        bq -= lr * bq.grad
    w2.grad.zero_(); w1.grad.zero_(); bq.grad.zero_()
    if epoch % 500 == 0 or epoch == 1:
        print(f"[P1(b)] epoch {epoch:4d} loss={loss.item():.4f} w2={w2.item():.4f} w1={w1.item():.4f} b={bq.item():.4f}")

# (c) Plot: show both fits together 
xs = torch.linspace(float(x.min()), float(x.max()), 100)
with torch.no_grad():
    y_lin = w_lin*xs + b_lin
    y_quad = w2*(xs**2) + w1*xs + bq

plt.figure()
plt.scatter(x.numpy(), y.numpy(), label="data")
plt.plot(xs.numpy(), y_lin.numpy(), label="linear")
plt.plot(xs.numpy(), y_quad.numpy(), label="quadratic")
plt.xlabel("measurement (normalized)"); plt.ylabel("temperature (°C)")
plt.title("HW5 Problem 1 — Linear vs Quadratic (vanilla GD)")
plt.legend()
plt.savefig("p1_fit.png", bbox_inches="tight")
print("Saved plot: p1_fit.png")
