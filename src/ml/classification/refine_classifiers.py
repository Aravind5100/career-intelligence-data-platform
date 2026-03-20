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

RANDOM_STATE = 42


def build_mixed_preprocessor():
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
                    max_features=25000,
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


def build_text_only_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=25000,
                    ngram_range=(1, 2),
                    min_df=5
                ),
                "text_blob"
            )
        ],
        remainder="drop"
    )


def prepare_data():
    print("Reading classification feature table...")
    df = pd.read_csv(INPUT_PATH)

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
        df[col] = df[col].map({True: 1, False: 0})

    df["salary_year_avg"] = pd.to_numeric(df["salary_year_avg"], errors="coerce")
    df["salary_hour_avg"] = pd.to_numeric(df["salary_hour_avg"], errors="coerce")

    print("Using full dataset for refinement.")
    print("Modeling shape:", df.shape)

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
    # Model 1: Improved Logistic Regression
    # ---------------------------------------------------
    print("\nTraining refined Logistic Regression...")
    logistic_pipeline = Pipeline(steps=[
        ("prep", build_mixed_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=4000,
            solver="saga",
            class_weight="balanced"
        ))
    ])

    logistic_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(
        "logistic_regression_refined",
        logistic_pipeline,
        X_test,
        y_test
    ))

    # ---------------------------------------------------
    # Model 2: Text-only Linear SVM
    # ---------------------------------------------------
    print("Training text-only Linear SVM...")
    svm_pipeline = Pipeline(steps=[
        ("prep", build_text_only_preprocessor()),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    svm_pipeline.fit(X_train[["text_blob"]], y_train)
    results.append(evaluate_model(
        "linear_svm_text_only",
        svm_pipeline,
        X_test[["text_blob"]],
        y_test
    ))

    # ---------------------------------------------------
    # Model 3: Improved Random Forest with richer SVD
    # ---------------------------------------------------
    print("Training refined Random Forest with SVD...")
    rf_pipeline = Pipeline(steps=[
        ("prep", build_mixed_preprocessor()),
        ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=250,
            max_depth=35,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(
        "random_forest_svd_refined",
        rf_pipeline,
        X_test,
        y_test
    ))

    # ---------------------------------------------------
    # Save comparison
    # ---------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    comparison_path = OUTPUT_DIR / "classification_model_comparison_refined.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\nRefined model comparison:")
    print(results_df)

    best_model_name = results_df.iloc[0]["model_name"]
    print(f"\nBest refined model: {best_model_name}")

    if best_model_name == "logistic_regression_refined":
        best_model = logistic_pipeline
    elif best_model_name == "linear_svm_text_only":
        best_model = svm_pipeline
    else:
        best_model = rf_pipeline

    model_path = MODEL_DIR / "best_role_classifier_refined.joblib"
    joblib.dump(best_model, model_path)

    print(f"Saved refined best model to: {model_path}")
    print(f"Saved refined comparison table to: {comparison_path}")


if __name__ == "__main__":
    main()