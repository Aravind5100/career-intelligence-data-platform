import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/ml_ready/feature_job_anomalies.csv")
EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = EXPORT_DIR / "ml_posting_anomalies.csv"


def determine_reason(row):
    reasons = []

    if row["role_disagreement_flag"] == 1:
        reasons.append("Role disagreement")

    if row["salary_anomaly_flag"] == 1:
        reasons.append("Salary outlier")

    if row["skill_count_anomaly_flag"] == 1:
        reasons.append("Skill count outlier")

    if len(reasons) == 0:
        return "Model-detected anomaly"

    if len(reasons) == 1:
        return reasons[0]

    return " + ".join(reasons)


def main():
    print("Reading anomaly feature table...")
    df = pd.read_csv(INPUT_PATH)

    # -----------------------------
    # Select modeling features
    # -----------------------------
    model_df = df[[
        "role_disagreement_flag",
        "salary_ratio_to_role_median",
        "job_skill_count",
        "skill_count_ratio_to_role_median",
        "salary_anomaly_flag",
        "skill_count_anomaly_flag"
    ]].copy()

    # Replace inf values if any
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    # Impute missing numeric values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(model_df)

    # -----------------------------
    # Isolation Forest
    # -----------------------------
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    iso.fit(X)

    # Scores and labels
    df["anomaly_score"] = iso.decision_function(X)
    df["anomaly_pred"] = iso.predict(X)  # -1 = anomaly, 1 = normal

    df["anomaly_label"] = df["anomaly_pred"].map({
        -1: "Anomalous",
         1: "Normal"
    })

    # -----------------------------
    # Add human-readable reason
    # -----------------------------
    anomaly_df = df[df["anomaly_label"] == "Anomalous"].copy()
    anomaly_df["anomaly_reason"] = anomaly_df.apply(determine_reason, axis=1)

    # Sort most anomalous first
    anomaly_df = anomaly_df.sort_values("anomaly_score", ascending=True)

    output_cols = [
        "job_key",
        "job_title",
        "company_name",
        "job_location",
        "actual_role",
        "predicted_role",
        "anomaly_score",
        "anomaly_label",
        "anomaly_reason",
        "role_disagreement_flag",
        "salary_anomaly_flag",
        "skill_count_anomaly_flag",
        "salary_year_avg",
        "role_median_salary_year",
        "job_skill_count",
        "role_median_skill_count"
    ]

    anomaly_df[output_cols].to_csv(OUTPUT_PATH, index=False)

    print("\nAnomaly detection complete.")
    print("Output path:", OUTPUT_PATH)
    print("Anomalous rows:", anomaly_df.shape[0])

    print("\nAnomaly reason counts:")
    print(anomaly_df["anomaly_reason"].value_counts().head(15))

    print("\nTop anomalous postings:")
    print(anomaly_df[[
        "job_title",
        "actual_role",
        "predicted_role",
        "anomaly_score",
        "anomaly_reason"
    ]].head(15))


if __name__ == "__main__":
    main()