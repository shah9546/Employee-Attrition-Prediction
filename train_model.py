import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("employee_attrition.csv")


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


# ============================================
# 3. FEATURES AND TARGET
# ============================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# ============================================
# 4. IDENTIFY COLUMNS
# ============================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ============================================
# 5. PREPROCESSING
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
# 7. CREATE MODEL PIPELINE
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================
# 8. TRAIN MODEL
# ============================================

print("Training Logistic Regression model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================
# 9. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 10. MODEL EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stay", "Leave"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)