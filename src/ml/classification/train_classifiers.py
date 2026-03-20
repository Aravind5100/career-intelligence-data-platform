import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/ml_ready/feature_role_classification.csv")
OUTPUT_DIR = Path("outputs/ml")
MODEL_DIR = Path("models/classification")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Settings
# -----------------------------
MAX_SAMPLE_ROWS = 150_000
RANDOM_STATE = 42


def build_preprocessor():
    text_feature = "text_blob"
    numeric_features = [
        "salary_year_avg",
        "salary_hour_avg",
        "job_work_from_home",
        "job_no_degree_mention",
        "job_health_insurance"
    ]

    categorical_features = ["salary_rate"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=5
                ),
                text_feature
            ),
            (
                "num",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_features
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_data():
    print("Reading classification feature table...")
    df = pd.read_csv(INPUT_PATH)

    # Fix boolean columns from CSV and convert to numeric 0/1
    bool_cols = [
        "job_work_from_home",
        "job_no_degree_mention",
        "job_health_insurance"
    ]

    for col in bool_cols:
        df[col] = df[col].replace({
            "True": True,
            "False": False,
            "true": True,
            "false": False
        })

    # Convert boolean values to numeric 0/1 while preserving missing values
    df[col] = df[col].map({True: 1, False: 0})

    # Numeric columns
    df["salary_year_avg"] = pd.to_numeric(df["salary_year_avg"], errors="coerce")
    df["salary_hour_avg"] = pd.to_numeric(df["salary_hour_avg"], errors="coerce")

    print("Original shape:", df.shape)

    print("Using full dataset for training benchmark.")
    print("Modeling shape:", df.shape)
    print("\nClass distribution in modeling dataset:")
    print(df["job_title_short"].value_counts())

    X = df[[
        "text_blob",
        "job_work_from_home",
        "job_no_degree_mention",
        "job_health_insurance",
        "salary_year_avg",
        "salary_hour_avg",
        "salary_rate"
    ]].copy()

    y = df["job_title_short"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    report = classification_report(y_test, preds, zero_division=0)

    report_path = OUTPUT_DIR / f"classification_report_{name}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4)
    }


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    results = []

    # ---------------------------------------------------
    # Model 1: Logistic Regression
    # ---------------------------------------------------
    print("\nTraining Logistic Regression...")
    logistic_pipeline = Pipeline(steps=[
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    logistic_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(
        "logistic_regression",
        logistic_pipeline,
        X_test,
        y_test
    ))

    # ---------------------------------------------------
    # Model 2: Linear SVM
    # ---------------------------------------------------
    print("Training Linear SVM...")
    svm_pipeline = Pipeline(steps=[
        ("prep", build_preprocessor()),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    svm_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(
        "linear_svm",
        svm_pipeline,
        X_test,
        y_test
    ))

    # ---------------------------------------------------
    # Model 3: Random Forest with SVD reduction
    # ---------------------------------------------------
    print("Training Random Forest with SVD...")
    rf_pipeline = Pipeline(steps=[
        ("prep", build_preprocessor()),
        ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(
        "random_forest_svd",
        rf_pipeline,
        X_test,
        y_test
    ))

    # ---------------------------------------------------
    # Save model comparison
    # ---------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    comparison_path = OUTPUT_DIR / "classification_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\nModel comparison:")
    print(results_df)

    # Save best model
    best_model_name = results_df.iloc[0]["model_name"]
    print(f"\nBest model: {best_model_name}")

    if best_model_name == "logistic_regression":
        best_model = logistic_pipeline
    elif best_model_name == "linear_svm":
        best_model = svm_pipeline
    else:
        best_model = rf_pipeline

    model_path = MODEL_DIR / "best_role_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")
    print(f"Saved comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()