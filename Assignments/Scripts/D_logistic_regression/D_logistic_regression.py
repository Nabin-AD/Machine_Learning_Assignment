"""
Question Set D - Logistic Regression
Dataset: heart_disease.csv
Implemented using ONLY NumPy and Pandas (no scikit-learn).
Trained via batch gradient descent on the log-loss.
"""
import numpy as np
import pandas as pd

np.random.seed(1)

# 1. Load dataset
df = pd.read_csv("../data/heart_disease.csv")
print("=== Dataset (head) ===")
print(df.head(), "\n")

# 2. Feature matrix and target
feature_cols = ["Age", "Cholesterol", "BloodPressure", "MaxHeartRate"]
X = df[feature_cols].values.astype(float)
y = df["Target"].values.astype(float)

# Standardize features manually (helps gradient descent converge)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# 3. Train-test split (80/20) manually
n = len(X_scaled)
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
X_test_b = np.column_stack([np.ones(len(X_test)), X_test])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 4. Train Logistic Regression classifier via gradient descent
n_features = X_train_b.shape[1]
weights = np.zeros(n_features)
lr = 0.1
epochs = 3000
m = len(y_train)

for epoch in range(epochs):
    z = X_train_b @ weights
    preds = sigmoid(z)
    gradient = (X_train_b.T @ (preds - y_train)) / m
    weights -= lr * gradient

print("=== Learned weights (bias, Age, Cholesterol, BloodPressure, MaxHeartRate) ===")
print(weights, "\n")

# 5. Predict class labels and probabilities for test data
probs_test = sigmoid(X_test_b @ weights)
y_pred = (probs_test >= 0.5).astype(int)

print("=== Predicted class probabilities (test set) ===")
for p, c in zip(probs_test, y_pred):
    print(f"P(disease)={p:.4f}  -> Predicted class: {c}")

# 6. Confusion matrix computed manually
TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

print("\n=== Confusion Matrix ===")
print(f"              Predicted 0   Predicted 1")
print(f"Actual 0         {TN:^10}    {FP:^10}")
print(f"Actual 1         {FN:^10}    {TP:^10}")

# 7. Accuracy, Precision, Recall, F1-score computed manually
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

print("\n=== Performance Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
