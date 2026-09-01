"""
Question Set G - Decision Tree
Dataset: loan_approval.csv
Scikit-learn is used here (as permitted for this experiment).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/loan_approval.csv")
print("=== Dataset (head) ===")
print(df.head(), "\n")

X = df.drop(columns=["Approved"])
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Decision Tree using Gini Index
tree_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_gini.fit(X_train, y_train)
pred_gini = tree_gini.predict(X_test)
acc_gini = accuracy_score(y_test, pred_gini)

# 2. Decision Tree using Entropy
tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_entropy.fit(X_train, y_train)
pred_entropy = tree_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, pred_entropy)

# 3. Compare classification accuracy
print("=== Accuracy Comparison ===")
print(f"Gini Index Decision Tree Accuracy:  {acc_gini:.4f}")
print(f"Entropy Decision Tree Accuracy:     {acc_entropy:.4f}")

# 4. Feature importance
print("\n=== Feature Importance (Gini model) ===")
for name, importance in zip(X.columns, tree_gini.feature_importances_):
    print(f"{name}: {importance:.4f}")

# 5. Visualize the generated Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(
    tree_gini,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (Gini Index) - Loan Approval")
plt.tight_layout()
plt.savefig("../data/G_decision_tree.png", dpi=120)
print("\nDecision tree visualization saved as G_decision_tree.png")
