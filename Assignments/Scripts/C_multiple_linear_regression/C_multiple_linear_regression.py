"""
Question Set C - Multiple Linear Regression
Dataset: car_performance.csv
Implemented using ONLY NumPy and Pandas (no scikit-learn).
Solved via the Normal Equation: theta = (X^T X)^-1 X^T y
"""
import numpy as np
import pandas as pd

np.random.seed(0)

# 1. Load dataset
df = pd.read_csv("../data/car_performance.csv")
print("=== Dataset (head) ===")
print(df.head(), "\n")

# 2 & 3. Select input features and target
X = df[["Engine_Size_L", "Horsepower"]].values.astype(float)
y = df["Price_USD"].values.astype(float)

# 4. Train-test split (80/20) manually
n = len(X)
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 5. Train Multiple Linear Regression via Normal Equation
X_train_b = np.column_stack([np.ones(len(X_train)), X_train])  # add intercept column
theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
intercept = theta[0]
coefficients = theta[1:]

# 6. Display coefficients and intercept
print("=== Regression coefficients ===")
for name, coef in zip(["Engine_Size_L", "Horsepower"], coefficients):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}\n")

# 7. Predict on test set
X_test_b = np.column_stack([np.ones(len(X_test)), X_test])
y_pred = X_test_b @ theta

print("=== Predicted selling prices (test set) ===")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}   Predicted: {pred:.2f}")

# 8. R2 Score computed manually
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\nModel Accuracy (R2 Score): {r2:.4f}")
