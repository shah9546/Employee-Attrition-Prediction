import pandas as pd

# Load dataset
df = pd.read_csv("employee_attrition.csv")

print("=" * 60)
print("EMPLOYEE ATTRITION DATASET ANALYSIS")
print("=" * 60)

# Dataset shape
print("\n1. Dataset Shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Data types
print("\n2. Data Types:")
print(df.dtypes)

# Missing values
print("\n3. Missing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\n4. Duplicate Rows:")
print(df.duplicated().sum())

# Attrition distribution
print("\n5. Attrition Distribution:")
print(df["Attrition"].value_counts())

# Attrition percentage
print("\n6. Attrition Percentage:")
print(df["Attrition"].value_counts(normalize=True) * 100)

# Numerical summary
print("\n7. Numerical Data Summary:")
print(df.describe())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)