import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, log_loss

# === Load built-in Breast Cancer dataset ===
data = load_breast_cancer()
X, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_report(model, label):
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(Xtr, ytr)

    fracs = np.linspace(0.1, 1.0, 10)
    losses, accs = [], []
    for f in fracs:
        n = max(10, int(len(Xtr) * f))
        pipe.fit(Xtr[:n], ytr[:n])
        p = pipe.predict_proba(Xte)[:, 1]
        losses.append(log_loss(yte, p))
        accs.append(accuracy_score(yte, (p >= 0.5).astype(int)))

    yhat = pipe.predict(Xte)
    print(f"\n=== {label} ===")
    print("Accuracy:", accuracy_score(yte, yhat))
    print("Precision:", precision_score(yte, yhat))
    print("Recall:", recall_score(yte, yhat))
    print("F1:", f1_score(yte, yhat))

    plt.figure(); plt.plot(fracs, losses, marker='o'); plt.title(f"{label} – Log-Loss"); plt.xlabel("Train fraction"); plt.ylabel("Loss"); plt.grid(True)
    plt.figure(); plt.plot(fracs, accs, marker='o'); plt.title(f"{label} – Accuracy"); plt.xlabel("Train fraction"); plt.ylabel("Accuracy"); plt.grid(True)
    plt.figure(); ConfusionMatrixDisplay.from_predictions(yte, yhat); plt.title(f"{label} – Confusion Matrix")
    plt.show()

# Problem 2(a): No penalty
train_and_report(LogisticRegression(penalty='none', solver='lbfgs', max_iter=500), "Cancer – LogisticRegression (no penalty)")

# Problem 2(b): With L2 weight penalty
train_and_report(LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', max_iter=500), "Cancer – LogisticRegression (L2, C=0.5)")
