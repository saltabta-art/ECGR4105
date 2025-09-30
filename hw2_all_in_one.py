# HW2 Linear Regression with Gradient Descent (All-in-One)
# Problems 1a/1b (no scaling), 2a/2b (feature scaling only), 3a/3b (scaling + L2)
# Target variable (Price) remains UNscaled throughout.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- Utilities -----------------
def seed_split(n, seed=42, frac=0.8):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(frac*n)
    return idx[:cut], idx[cut:]

def build_xy(df, feats):
    X = df[feats].to_numpy(float)
    y = df['price'].to_numpy(float).reshape(-1,1)  # target unscaled
    return X, y

def add_bias(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def mse(y,yh):
    return float(((y-yh)**2).mean())

def standardize_fit(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, ddof=0, keepdims=True); sd[sd==0]=1.0
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X-mu)/sd

def normalize_fit(X):
    mn = X.min(0, keepdims=True); mx = X.max(0, keepdims=True)
    sp = mx - mn; sp[sp==0]=1.0
    return mn, sp

def normalize_apply(X, mn, sp):
    return (X-mn)/sp

def gd_backtracking(Xtr, ytr, Xva, yva, n_iters=1500, alpha0=1.0, beta=0.5, c=1e-4, l2=0.0):
    # Our own GD with Armijo backtracking. l2 excludes bias (index 0).
    m, n = Xtr.shape
    th = np.zeros((n,1))
    I = np.eye(n); I[0,0]=0.0
    tr_hist, va_hist = [], []
    for t in range(n_iters):
        r = Xtr @ th - ytr
        g = (2.0/m)*(Xtr.T @ r) + (2.0*l2/m)*(I @ th)
        g2 = float((g*g).sum())
        step = alpha0
        f0 = mse(ytr, Xtr@th) + (l2/m)*float((th[1:]**2).sum())
        while True:
            th_new = th - step*g
            f_new = mse(ytr, Xtr@th_new) + (l2/m)*float((th_new[1:]**2).sum())
            if f_new <= f0 - c*step*g2:
                th = th_new; break
            step *= beta
            if step < 1e-20:
                th = th_new; break
        if t==0 or (t+1)%10==0 or t==n_iters-1:
            tr_hist.append(mse(ytr, Xtr@th))
            va_hist.append(mse(yva, Xva@th))
    return th, np.array(tr_hist), np.array(va_hist)

def save_plot(xlabel, ylabel, title, curves, labels, outpath):
    plt.figure()
    for c,lab in zip(curves, labels):
        plt.plot(c, label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.legend()
    plt.savefig(outpath, bbox_inches='tight'); plt.close()
    return outpath

def ensure_bool(df, cols):
    for c in cols:
        df[c] = df[c].map({'yes':1,'no':0}) if df[c].dtype=='O' else df[c]
        df[c] = df[c].astype(int)

# ----------------- Load data -----------------
df = pd.read_csv('Housing.csv')
df.columns = [c.lower() for c in df.columns]
ensure_bool(df, ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea'])
train_idx, val_idx = seed_split(len(df), seed=42, frac=0.8)

# Feature sets
f1a = ['area','bedrooms','bathrooms','stories','parking']
f1b = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement',
       'hotwaterheating','airconditioning','parking','prefarea']

# Directory for outputs
outdir = Path('hw2_outputs'); outdir.mkdir(exist_ok=True)

# ----------------- Problem 1a (no scaling) -----------------
X, y = build_xy(df, f1a)
Xtr, ytr, Xva, yva = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
th_1a, tr_1a, va_1a = gd_backtracking(add_bias(Xtr), ytr, add_bias(Xva), yva, n_iters=3000)
save_plot('Checkpoint (every 10 iters)','MSE','P1a — No scaling (GD+backtracking)',
          [tr_1a, va_1a], ['Train','Validation'], outdir/'P1a.png')
pd.Series(th_1a.flatten(), index=['bias']+f1a).to_csv(outdir/'theta_1a.csv')

# ----------------- Problem 1b (no scaling) -----------------
Xb, yb = build_xy(df, f1b)
Xtrb, ytrb, Xvab, yvab = Xb[train_idx], yb[train_idx], Xb[val_idx], yb[val_idx]
th_1b, tr_1b, va_1b = gd_backtracking(add_bias(Xtrb), ytrb, add_bias(Xvab), yvab, n_iters=3000)
save_plot('Checkpoint (every 10 iters)','MSE','P1b — No scaling (GD+backtracking)',
          [tr_1b, va_1b], ['Train','Validation'], outdir/'P1b.png')
pd.Series(th_1b.flatten(), index=['bias']+f1b).to_csv(outdir/'theta_1b.csv')

# ----------------- Problem 2a (scaling inputs only) -----------------
mu, sd = standardize_fit(Xtr); Xtr_std, Xva_std = standardize_apply(Xtr,mu,sd), standardize_apply(Xva,mu,sd)
th_2a_std, tr_2a_std, va_2a_std = gd_backtracking(add_bias(Xtr_std), ytr, add_bias(Xva_std), yva, n_iters=1500)
mn, sp = normalize_fit(Xtr); Xtr_n, Xva_n = normalize_apply(Xtr,mn,sp), normalize_apply(Xva,mn,sp)
th_2a_n, tr_2a_n, va_2a_n = gd_backtracking(add_bias(Xtr_n), ytr, add_bias(Xva_n), yva, n_iters=1500)
save_plot('Checkpoint (every 10 iters)','MSE','P2a — Std vs Norm (inputs only)',
          [tr_2a_std, va_2a_std, tr_2a_n, va_2a_n],
          ['Train std','Val std','Train norm','Val norm'], outdir/'P2a_compare.png')

# ----------------- Problem 2b (scaling inputs only) -----------------
mu_b, sd_b = standardize_fit(Xtrb); Xtrb_std, Xvab_std = standardize_apply(Xtrb,mu_b,sd_b), standardize_apply(Xvab,mu_b,sd_b)
th_2b_std, tr_2b_std, va_2b_std = gd_backtracking(add_bias(Xtrb_std), ytrb, add_bias(Xvab_std), yvab, n_iters=1500)
mn_b, sp_b = normalize_fit(Xtrb); Xtrb_n, Xvab_n = normalize_apply(Xtrb,mn_b,sp_b), normalize_apply(Xvab,mn_b,sp_b)
th_2b_n, tr_2b_n, va_2b_n = gd_backtracking(add_bias(Xtrb_n), ytrb, add_bias(Xvab_n), yvab, n_iters=1500)
save_plot('Checkpoint (every 10 iters)','MSE','P2b — Std vs Norm (inputs only)',
          [tr_2b_std, va_2b_std, tr_2b_n, va_2b_n],
          ['Train std','Val std','Train norm','Val norm'], outdir/'P2b_compare.png')

# ----------------- Problem 3a (scaling + L2) -----------------
th_3a, tr_3a, va_3a = gd_backtracking(add_bias(Xtr_n), ytr, add_bias(Xva_n), yva, n_iters=1500, l2=1.0)
save_plot('Checkpoint (every 10 iters)','MSE','P3a — Norm + L2 (λ=1.0)',
          [tr_3a, va_3a], ['Train','Validation'], outdir/'P3a.png')
pd.Series(th_3a.flatten(), index=['bias']+f1a).to_csv(outdir/'theta_3a.csv')

# ----------------- Problem 3b (scaling + L2) -----------------
th_3b, tr_3b, va_3b = gd_backtracking(add_bias(Xtrb_n), ytrb, add_bias(Xvab_n), yvab, n_iters=1500, l2=1.0)
save_plot('Checkpoint (every 10 iters)','MSE','P3b — Norm + L2 (λ=1.0)',
          [tr_3b, va_3b], ['Train','Validation'], outdir/'P3b.png')
pd.Series(th_3b.flatten(), index=['bias']+f1b).to_csv(outdir/'theta_3b.csv')

# ----------------- Print summary -----------------
def last(a): return float(a[-1]) if len(a)>0 else float('nan')
print('Final Validation MSEs:')
print('  1a:', last(va_1a))
print('  1b:', last(va_1b))
print('  2a (std):', last(va_2a_std), '  2a (norm):', last(va_2a_n))
print('  2b (std):', last(va_2b_std), '  2b (norm):', last(va_2b_n))
print('  3a (norm+L2):', last(va_3a))
print('  3b (norm+L2):', last(va_3b))

print("\nSaved outputs to:", Path(outdir).resolve())
