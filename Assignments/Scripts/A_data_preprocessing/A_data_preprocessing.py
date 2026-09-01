"""
Question Set A - Data Preprocessing
Dataset: employee_data.csv
Implemented using ONLY Pandas and NumPy (no scikit-learn).
"""
import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("../data/employee_data.csv")

# 2. First five records
print("=== First 5 records ===")
print(df.head(), "\n")

# 3. Identify missing values
print("=== Missing value summary ===")
print(df.isnull().sum(), "\n")

# Separate numerical and categorical columns (excluding ID)
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "EmployeeID"]
categorical_cols = df.select_dtypes(include="object").columns.tolist()

# 4. Mean Imputation for numerical columns
df_imputed = df.copy()
for col in numeric_cols:
    mean_val = df_imputed[col].mean()
    df_imputed[col] = df_imputed[col].fillna(mean_val)

# 5. Mode Imputation for categorical columns
for col in categorical_cols:
    mode_val = df_imputed[col].mode()[0]
    df_imputed[col] = df_imputed[col].fillna(mode_val)

print("=== Dataset after imputation ===")
print(df_imputed.head(), "\n")

# 6. One-Hot Encoding for 'Department' using pure pandas (get_dummies)
df_encoded = pd.get_dummies(df_imputed, columns=["Department"], prefix="Dept")
# Convert bool dummy columns to int (0/1)
dummy_cols = [c for c in df_encoded.columns if c.startswith("Dept_")]
df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(int)

print("=== Encoded feature columns ===")
print(df_encoded[dummy_cols].head(), "\n")

# 7. Standardize numerical features manually: z = (x - mean) / std
df_scaled = df_encoded.copy()
for col in numeric_cols:
    mean = df_scaled[col].mean()
    std = df_scaled[col].std(ddof=0)
    df_scaled[col] = (df_scaled[col] - mean) / std

print("=== Scaled numerical values ===")
print(df_scaled[numeric_cols].head(), "\n")

# 8. Final transformed dataset and dimensions
print("=== Final transformed dataset ===")
print(df_scaled.head())
print("\nDimensions after preprocessing:", df_scaled.shape)

df_scaled.to_csv("../data/employee_data_transformed.csv", index=False)
print("\nSaved transformed dataset to employee_data_transformed.csv")
