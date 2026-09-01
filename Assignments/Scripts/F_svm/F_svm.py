"""
Question Set F - Support Vector Machine
Dataset: synthetic_moons.csv
Implemented using ONLY NumPy and Pandas (no scikit-learn).

Both the Linear-kernel and RBF-kernel SVMs are trained with the
Kernel Pegasos algorithm (Shalev-Shwartz et al.), a stochastic
sub-gradient method for the SVM hinge-loss objective that works
directly with a kernel matrix - no external ML library required.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(3)

# Load dataset
df = pd.read_csv("../data/synthetic_moons.csv")
X = df[["X1", "X2"]].values.astype(float)
y = df["Label"].values.astype(float)
y = np.where(y == 0, -1, 1)  # SVM uses {-1, +1} labels

# Train-test split (80/20)
n = len(X)
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


def linear_kernel(A, B):
    return A @ B.T


def rbf_kernel(A, B, gamma=1.0):
    A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
    B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
    sq_dists = A_sq + B_sq - 2 * A @ B.T
    return np.exp(-gamma * sq_dists)


def kernel_pegasos(X_train, y_train, kernel_fn, lam=0.01, T=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X_train)
    alpha = np.zeros(n)
    K = kernel_fn(X_train, X_train)
    for t in range(1, T + 1):
        i = rng.integers(0, n)
        decision = (alpha * y_train) @ K[:, i] / (lam * t)
        if y_train[i] * decision < 1:
            alpha[i] += 1
    return alpha, K


def kernel_decision_function(alpha, y_train, X_train, X_query, kernel_fn, lam, T):
    K_query = kernel_fn(X_train, X_query)  # (n_train, n_query)
    scores = (alpha * y_train) @ K_query / (lam * T)
    return scores


def evaluate(alpha, y_train, X_train, X_test, y_test, kernel_fn, lam, T):
    scores = kernel_decision_function(alpha, y_train, X_train, X_test, kernel_fn, lam, T)
    preds = np.where(scores >= 0, 1, -1)
    acc = np.mean(preds == y_test)
    return acc, preds


LAM = 0.01
T = 3000

# 1. Train Linear-kernel SVM
alpha_lin, _ = kernel_pegasos(X_train, y_train, linear_kernel, lam=LAM, T=T, seed=10)
acc_lin, pred_lin = evaluate(alpha_lin, y_train, X_train, X_test, y_test, linear_kernel, LAM, T)
sv_lin = np.sum(alpha_lin > 0)

# 2. Train RBF-kernel SVM
GAMMA = 2.0
rbf = lambda A, B: rbf_kernel(A, B, gamma=GAMMA)
alpha_rbf, _ = kernel_pegasos(X_train, y_train, rbf, lam=LAM, T=T, seed=20)
acc_rbf, pred_rbf = evaluate(alpha_rbf, y_train, X_train, X_test, y_test, rbf, LAM, T)
sv_rbf = np.sum(alpha_rbf > 0)

# 3. Compare classification accuracies
print("=== Classification Accuracy Comparison ===")
print(f"Linear Kernel SVM Accuracy: {acc_lin:.4f}")
print(f"RBF Kernel SVM Accuracy:    {acc_rbf:.4f}")

# 4. Number of support vectors
print("\n=== Support Vectors ===")
print(f"Linear Kernel -> Support Vectors: {sv_lin} / {len(X_train)}")
print(f"RBF Kernel    -> Support Vectors: {sv_rbf} / {len(X_train)}")

# 5. Plot decision boundaries for both classifiers
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
grid = np.column_stack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (alpha, kernel_fn, title) in zip(
    axes,
    [(alpha_lin, linear_kernel, f"Linear Kernel (acc={acc_lin:.2f}, SV={sv_lin})"),
     (alpha_rbf, rbf, f"RBF Kernel (acc={acc_rbf:.2f}, SV={sv_rbf})")]
):
    Z = kernel_decision_function(alpha, y_train, X_train, grid, kernel_fn, LAM, T)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#FFD9D9", "#D9E8FF"], alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="blue", edgecolor="k", label="Class 1")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c="red", edgecolor="k", label="Class 0")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig("../data/F_svm_decision_boundaries.png", dpi=120)
print("\nDecision boundary plot saved as F_svm_decision_boundaries.png")
