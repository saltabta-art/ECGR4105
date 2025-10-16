import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, log_loss

# === Load your diabetes.csv ===
df = pd.read_csv('diabetes.csv')

# Detect label column automatically
possible_labels = ['Outcome', 'target', 'Label', 'class']
label_col = next((c for c in df.columns if c in possible_labels), df.columns[-1])

X = df.drop(columns=[label_col]).values
y = df[label_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Logistic Regression (via SGD) ===
clf = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='optimal', random_state=42, warm_start=True)
pipe = make_pipeline(StandardScaler(), clf)

epochs = 30
losses, accuracies = [], []

for epoch in range(epochs):
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    losses.append(log_loss(y_test, y_proba))
    accuracies.append(accuracy_score(y_test, (y_proba >= 0.5).astype(int)))

y_pred = pipe.predict(X_test)

print("=== Diabetes Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

plt.figure(); plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title("Diabetes – Test Log-Loss per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Log-Loss"); plt.grid(True)

plt.figure(); plt.plot(range(1, epochs + 1), accuracies, marker='o')
plt.title("Diabetes – Test Accuracy per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True)

plt.figure(); ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Diabetes – Confusion Matrix")
plt.show()
