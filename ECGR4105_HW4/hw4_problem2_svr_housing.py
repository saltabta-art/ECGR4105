# ECGR4105 - HW4 Problem 2 (Final, no 'squared', no dtype warnings)
# Support Vector Regression (SVR) on the Housing dataset (80/20 split)
# Run: python hw4_problem2_svr_housing.py

import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load dataset

df = pd.read_csv("Housing.csv")

# Convert Yes/No columns to 1/0 (leave as ints)
yn_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for c in yn_cols:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].str.strip().str.lower().map({'yes': 1, 'no': 0}).astype("Int64").astype(int)


# 2) Select features and target

feature_cols = ['area','bedrooms','bathrooms','stories','mainroad','guestroom',
                'basement','hotwaterheating','airconditioning','parking','prefarea']
target_col = 'price'

X = df[feature_cols].copy()
y = df[target_col].copy()


# 3) Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Scale numeric columns (cast to float first to avoid dtype warnings)

numeric_cols = ['area','bedrooms','bathrooms','stories','parking']

# Ensure float dtype BEFORE scaling to avoid the FutureWarning
X_train[numeric_cols] = X_train[numeric_cols].astype(float)
X_test[numeric_cols]  = X_test[numeric_cols].astype(float)

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

# 5) Train and evaluate SVR kernels

kernels = ['linear', 'rbf', 'poly']
results = []

for k in kernels:
    svr = SVR(kernel=k, C=1.0, degree=3, gamma='scale')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    # RMSE without 'squared=False' (compatible with older sklearn)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((k, rmse, r2))

    # Plot actual vs predicted
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"SVR ({k}) - Actual vs Predicted")
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.tight_layout()
    plt.savefig(f"p2_actual_vs_pred_{k}.png", dpi=150)

# 6) Summary bar charts

kernels_list = [r[0] for r in results]
rmses = [r[1] for r in results]
r2s = [r[2] for r in results]

plt.figure(figsize=(7,5))
plt.bar(kernels_list, rmses)
plt.ylabel("RMSE (lower is better)")
plt.title("SVR Kernel Comparison - RMSE")
plt.tight_layout()
plt.savefig("p2_rmse_bar.png", dpi=150)

plt.figure(figsize=(7,5))
plt.bar(kernels_list, r2s)
plt.ylabel("R² (higher is better)")
plt.title("SVR Kernel Comparison - R²")
plt.tight_layout()
plt.savefig("p2_r2_bar.png", dpi=150)

# 7) Print results

print("=== SVR Results ===")
for k, rmse, r2 in results:
    print(f"{k:>6s} -> RMSE: {rmse:.2f}, R²: {r2:.3f}")
