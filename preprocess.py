import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("employee_attrition.csv")

print("Original dataset shape:", df.shape)


# ============================================
# 2. REMOVE UNNECESSARY COLUMNS
# ============================================

columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop)

print("After removing unnecessary columns:", df.shape)


# ============================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================

X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# ============================================
# 4. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================
# 5. CREATE PREPROCESSOR
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ============================================
# 6. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# 7. FIT AND TRANSFORM DATA
# ============================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ============================================
# 8. DISPLAY RESULTS
# ============================================

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nPreprocessing completed successfully!")