import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("employee_attrition.csv")

print("Dataset loaded:", df.shape)


# ============================================
# 2. REMOVE IRRELEVANT COLUMNS
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
# 4. IDENTIFY CATEGORICAL COLUMNS
# ============================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================
# 5. PREPROCESSOR
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
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
# 7. RANDOM FOREST
# ============================================

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================
# 8. COMPLETE PIPELINE
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ============================================
# 9. TRAIN
# ============================================

print("\nTraining final Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================
# 10. PREDICTION
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 11. EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stay", "Leave"]
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# ============================================
# 12. SAVE MODEL
# ============================================

joblib.dump(
    model,
    "employee_attrition_model.pkl"
)

print("\nModel saved successfully!")
print("File: employee_attrition_model.pkl")

print("\n" + "=" * 60)
print("FINAL MODEL READY")
print("=" * 60)