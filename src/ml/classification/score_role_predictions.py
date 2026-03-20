import pandas as pd
from pathlib import Path
import joblib

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/ml_ready/feature_role_classification.csv")
MODEL_PATH = Path("models/classification/best_role_classifier_refined.joblib")
EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = EXPORT_DIR / "ml_role_predictions.csv"


def main():
    print("Reading classification feature table...")
    df = pd.read_csv(INPUT_PATH)

    print("Loading best refined classifier...")
    model = joblib.load(MODEL_PATH)

    # Keep only the columns needed by the text-only SVM
    X = df[["text_blob"]].copy()

    print("Scoring predictions on full classification dataset...")
    df["predicted_role"] = model.predict(X)

    # Compare against actual label
    df["actual_role"] = df["job_title_short"]
    df["correct_prediction"] = df["actual_role"] == df["predicted_role"]

    # Keep useful export columns
    scored_output = df[[
        "job_key",
        "actual_role",
        "predicted_role",
        "correct_prediction"
    ]].copy()

    scored_output.to_csv(OUTPUT_PATH, index=False)

    print("\nScoring complete.")
    print("Output path:", OUTPUT_PATH)
    print("Output shape:", scored_output.shape)

    print("\nPrediction correctness summary:")
    print(scored_output["correct_prediction"].value_counts())

    print("\nTop actual vs predicted combinations:")
    print(
        scored_output.groupby(["actual_role", "predicted_role"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )


if __name__ == "__main__":
    main()