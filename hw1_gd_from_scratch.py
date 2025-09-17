import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = "D3.csv.xlsx"  # or "D3.csv"
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def load_dataset(path_str):
    p = Path(path_str)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, header=0)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p, header=0)
    else:
        raise ValueError("Use .xlsx/.xls or .csv")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: c.upper() for c in df.columns})
    wanted = ["X1","X2","X3","Y"]
    if not all(w in df.columns for w in wanted):
        df = df.iloc[:, :4].copy()
        df.columns = wanted
    for c in wanted:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=wanted).reset_index(drop=True)[wanted].copy()

def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2))

def gd_single(x, y, alpha=0.05, iters=5000):
    m = len(y)
    t0, t1 = 0.0, 0.0
    loss_hist = np.empty(iters, dtype=float)
    for k in range(iters):
        y_hat = t0 + t1 * x
        err = y_hat - y
        g0 = err.mean()
        g1 = (err * x).mean()
        t0 -= alpha * g0
        t1 -= alpha * g1
        loss_hist[k] = mse_loss(y, y_hat)
    return t0, t1, loss_hist

def gd_multi(X, y, alpha=0.05, iters=15000):
    m, n = X.shape
    Xb = np.hstack([np.ones((m,1)), X])
    theta = np.zeros(n+1)  # [t0, t1, t2, t3]
    loss_hist = np.empty(iters, dtype=float)
    for k in range(iters):
        y_hat = Xb @ theta
        err = y_hat - y
        grad = (Xb.T @ err) / m
        theta -= alpha * grad
        loss_hist[k] = mse_loss(y, y_hat)
    return theta, loss_hist

def predict_multi(theta, pts):
    pts = np.array(pts, dtype=float).reshape(-1,3)
    Xb = np.hstack([np.ones((pts.shape[0],1)), pts])
    return Xb @ theta

def plot_regression_fit(x, y, t0, t1, name):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(x, y)
    xs = np.linspace(x.min(), x.max(), 300)
    ys = t0 + t1 * xs
    ax.plot(xs, ys)
    ax.set_title(f"Regression Fit: {name} → Y")
    ax.set_xlabel(name); ax.set_ylabel("Y")
    ax.grid(True, linewidth=0.5)
    fig.savefig(PLOTS_DIR / f"fit_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_loss(loss_hist, title, filename):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(np.arange(1, len(loss_hist)+1), loss_hist)
    ax.set_title(title)
    ax.set_xlabel("Iteration"); ax.set_ylabel("MSE Loss")
    ax.grid(True, linewidth=0.5)
    fig.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    df = load_dataset(DATA_FILE)
    x1 = df["X1"].to_numpy(float)
    x2 = df["X2"].to_numpy(float)
    x3 = df["X3"].to_numpy(float)
    y  = df["Y"].to_numpy(float)
    X  = df[["X1","X2","X3"]].to_numpy(float)

    alphas = [0.10, 0.05, 0.02, 0.01]
    # Problem 1
    best_single = {}
    for name, x in [("X1", x1), ("X2", x2), ("X3", x3)]:
        best = None
        for a in alphas:
            t0, t1, L = gd_single(x, y, alpha=a, iters=5000)
            if best is None or L[-1] < best["final_loss"]:
                best = {"alpha": a, "t0": t0, "t1": t1, "loss_hist": L, "final_loss": L[-1]}
        best_single[name] = best
        plot_regression_fit(x, y, best["t0"], best["t1"], name)
        plot_loss(best["loss_hist"], f"Loss vs Iteration (best α={best['alpha']}) — {name}", f"loss_{name}.png")

    # Problem 2
    runs = []
    for a in alphas:
        theta, L = gd_multi(X, y, alpha=a, iters=15000)
        runs.append({"alpha": a, "theta": theta, "loss_hist": L, "final_loss": L[-1]})
    best_multi = min(runs, key=lambda r: r["final_loss"])
    th = best_multi["theta"]
    plot_loss(best_multi["loss_hist"], f"Multivariate Loss vs Iteration (α={best_multi['alpha']})",
              f"multi_loss_alpha_{str(best_multi['alpha']).replace('.','p')}.png")

    print("Problem 1 — Single-variable (best per feature):")
    for name in ["X1","X2","X3"]:
        b = best_single[name]
        print(f"  {name}: best α={b['alpha']}, y = {b['t0']:.10f} + {b['t1']:.10f}*{name}, final MSE={b['final_loss']:.10f}")

    print("\nProblem 2 — Multivariate:")
    print(f"  Best α={best_multi['alpha']}")
    print(f"  y = {th[0]:.10f} + {th[1]:.10f}*X1 + {th[2]:.10f}*X2 + {th[3]:.10f}*X3")
    print(f"  final MSE = {best_multi['final_loss']:.10f}")

    preds = predict_multi(th, [(1,1,1),(2,0,4),(3,2,1)])
    print("\nPredictions:")
    for pt, pv in zip([(1,1,1),(2,0,4),(3,2,1)], preds):
        print(f"  X={pt} → ŷ = {pv:.10f}")

    print(f"\nSaved plots to: {PLOTS_DIR.resolve()}")
