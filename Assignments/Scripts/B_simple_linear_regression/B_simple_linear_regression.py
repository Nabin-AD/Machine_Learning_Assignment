"""
Question Set B - Simple Linear Regression
Dataset: housing_price.csv
Implemented using ONLY NumPy and Pandas (no scikit-learn).
Regression solved via the closed-form Ordinary Least Squares (Normal Equation).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# 1. Load dataset
df = pd.read_csv("../data/housing_price.csv")
print("=== Dataset ===")
print(df.head(), "\n")

# 2. Independent and dependent variables
X = df["FloorArea_sqft"].values.astype(float)
y = df["Price"].values.astype(float)
print("Independent variable: FloorArea_sqft")
print("Dependent variable: Price\n")

# 3. Train-test split (80/20) done manually with NumPy
n = len(X)
indices = np.arange(n)
np.random.shuffle(indices)
split = int(0.8 * n)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}\n")

# 4. Train Simple Linear Regression: y = m*x + b  (least squares closed form)
x_mean, y_mean = X_train.mean(), y_train.mean()
m = np.sum((X_train - x_mean) * (y_train - y_mean)) / np.sum((X_train - x_mean) ** 2)
b = y_mean - m * x_mean

# 5. Slope and intercept
print(f"=== Regression equation ===")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Price = {m:.4f} * FloorArea + {b:.4f}\n")

# 6. Predictions on test set
y_pred = m * X_test + b
print("=== Predicted values (test set) ===")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}  Predicted: {pred:.2f}")

# 7. Error metrics computed manually
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("\n=== Error metrics ===")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 8. Plot regression line with original data points
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, color="steelblue", label="Training data")
plt.scatter(X_test, y_test, color="orange", label="Testing data")
x_line = np.linspace(X.min(), X.max(), 100)
plt.plot(x_line, m * x_line + b, color="red", linewidth=2, label="Regression line")
plt.xlabel("Floor Area (sqft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression: House Price vs Floor Area")
plt.legend()
plt.tight_layout()
plt.savefig("../data/B_regression_plot.png", dpi=120)
print("\nRegression plot saved as B_regression_plot.png")
