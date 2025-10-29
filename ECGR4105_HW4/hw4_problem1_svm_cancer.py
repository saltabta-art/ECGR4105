# ECGR4105 - HW4 Problem 1
# SVM classifier on Breast Cancer dataset (80/20 split)
# Saves plots and prints metrics for multiple kernels.
# Run: python hw4_problem1_svm_cancer.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def train_and_eval(kernels=('linear','rbf','poly','sigmoid'), C=1.0, degree=3, gamma='scale', random_state=42):
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 80/20 split as required
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scores = []
    reports = {}

    for k in kernels:
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=k, C=C, degree=degree, gamma=gamma, random_state=random_state))
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        scores.append((k, acc, prec, rec))
        reports[k] = classification_report(y_test, y_pred, target_names=data.target_names)

    return scores, reports

def plot_scores(scores, out_path="p1_metrics_bar.png"):
    kernels = [s[0] for s in scores]
    acc = [s[1] for s in scores]
    prec = [s[2] for s in scores]
    rec = [s[3] for s in scores]

    x = np.arange(len(kernels))
    width = 0.25

    plt.figure(figsize=(8,5))
    plt.bar(x - width, acc, width, label='Accuracy')
    plt.bar(x,         prec, width, label='Precision')
    plt.bar(x + width, rec,  width, label='Recall')
    plt.xticks(x, kernels)
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.title('SVM Kernels on Breast Cancer (80/20)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    scores, reports = train_and_eval()
    print("=== Metrics by Kernel ===")
    for k, acc, prec, rec in scores:
        print(f"{k:>7s} -> Acc: {acc:.3f}  Prec: {prec:.3f}  Recall: {rec:.3f}")
    print("\n=== Classification Reports ===")
    for k, rep in reports.items():
        print(f"\n--- {k} ---\n{rep}")
    plot_scores(scores)
