"""
Question Set E - K-Nearest Neighbour (KNN)
Dataset: customer_segments.csv
Implemented using ONLY NumPy and Pandas (no scikit-learn).
"""
import numpy as np
import pandas as pd

np.random.seed(2)

# 1. Load dataset
df = pd.read_csv("../data/customer_segments.csv")
print("=== Dataset (head) ===")
print(df.head(), "\n")

X = df[["AnnualIncome_k", "SpendingScore"]].values.astype(float)
y = df["Segment"].values.astype(int)

# 2. Standardize numerical features manually
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# Train-test split (80/20) manually
n = len(X_scaled)
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for point in X_test:
        # Euclidean distance to every training point
        distances = np.sqrt(np.sum((X_train - point) ** 2, axis=1))
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        # majority vote
        values, counts = np.unique(nearest_labels, return_counts=True)
        predictions.append(values[np.argmax(counts)])
    return np.array(predictions)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# 3 & 4. Train KNN classifier using k = 5 and predict
y_pred_k5 = knn_predict(X_train, y_train, X_test, k=5)
acc_k5 = accuracy(y_test, y_pred_k5)

print("=== Predictions (k=5) ===")
print(y_pred_k5)

# 5. Testing accuracy for k=5
print(f"\nTesting Accuracy (k=5): {acc_k5:.4f}")

# 6. Repeat for k=3 and compare
y_pred_k3 = knn_predict(X_train, y_train, X_test, k=3)
acc_k3 = accuracy(y_test, y_pred_k3)

print(f"Testing Accuracy (k=3): {acc_k3:.4f}")

print("\n=== Comparison ===")
print(f"k=3 -> Accuracy: {acc_k3:.4f}")
print(f"k=5 -> Accuracy: {acc_k5:.4f}")
better_k = 3 if acc_k3 > acc_k5 else (5 if acc_k5 > acc_k3 else "3 and 5 (tie)")
print(f"Better performing k on this test split: {better_k}")
