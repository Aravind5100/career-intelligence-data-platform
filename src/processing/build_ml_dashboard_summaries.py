import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROLE_PRED_PATH = Path("data/exports/ml_role_predictions.csv")
EMERGING_PATH = Path("data/exports/ml_emerging_skills.csv")
ANOMALY_PATH = Path("data/exports/ml_posting_anomalies.csv")
TRANSITION_PATH = Path("data/exports/ml_transition_recommendations.csv")

EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_SUMMARY_PATH = EXPORT_DIR / "ml_summary_classification.csv"
EMERGING_SUMMARY_PATH = EXPORT_DIR / "ml_summary_emerging_skills.csv"
ANOMALY_SUMMARY_PATH = EXPORT_DIR / "ml_summary_anomaly_counts.csv"
TRANSITION_SUMMARY_PATH = EXPORT_DIR / "ml_summary_top_transitions.csv"


def main():
    print("Reading ML outputs...")
    role_preds = pd.read_csv(ROLE_PRED_PATH)
    emerging = pd.read_csv(EMERGING_PATH)
    anomalies = pd.read_csv(ANOMALY_PATH)
    transitions = pd.read_csv(TRANSITION_PATH)

    # ---------------------------------------------------
    # 1. Classification summary
    # ---------------------------------------------------
    total_scored = len(role_preds)
    correct_count = role_preds["correct_prediction"].sum()
    incorrect_count = total_scored - correct_count
    agreement_rate = round(correct_count / total_scored, 4)

    top_confusions = (
        role_preds[role_preds["actual_role"] != role_preds["predicted_role"]]
        .groupby(["actual_role", "predicted_role"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    class_summary_rows = [{
        "metric": "total_scored_jobs",
        "value": total_scored
    }, {
        "metric": "correct_predictions",
        "value": int(correct_count)
    }, {
        "metric": "incorrect_predictions",
        "value": int(incorrect_count)
    }, {
        "metric": "agreement_rate",
        "value": agreement_rate
    }]

    # add top confusion pairs as extra rows
    for i, row in top_confusions.reset_index(drop=True).iterrows():
        class_summary_rows.append({
            "metric": f"top_confusion_{i+1}",
            "value": f"{row['actual_role']} -> {row['predicted_role']} ({row['count']})"
        })

    classification_summary = pd.DataFrame(class_summary_rows)
    classification_summary.to_csv(CLASS_SUMMARY_PATH, index=False)

    # ---------------------------------------------------
    # 2. Emerging skills summary
    # ---------------------------------------------------
    emerging_summary = emerging.sort_values("emerging_score", ascending=False).copy()
    emerging_summary.to_csv(EMERGING_SUMMARY_PATH, index=False)

    # ---------------------------------------------------
    # 3. Anomaly summary
    # ---------------------------------------------------
    anomaly_reason_counts = (
        anomalies["anomaly_reason"]
        .value_counts()
        .reset_index()
    )
    anomaly_reason_counts.columns = ["anomaly_reason", "count"]

    anomaly_summary_rows = [{
        "metric": "total_anomalies",
        "value": len(anomalies)
    }, {
        "metric": "unique_anomaly_reasons",
        "value": anomalies["anomaly_reason"].nunique()
    }]

    for _, row in anomaly_reason_counts.iterrows():
        anomaly_summary_rows.append({
            "metric": f"reason::{row['anomaly_reason']}",
            "value": int(row["count"])
        })

    anomaly_summary = pd.DataFrame(anomaly_summary_rows)
    anomaly_summary.to_csv(ANOMALY_SUMMARY_PATH, index=False)

    # ---------------------------------------------------
    # 4. Transition summary
    # ---------------------------------------------------
    # Keep strongest 3 target transitions per source role
    top_transitions = (
        transitions.sort_values(["source_role", "similarity_score"], ascending=[True, False])
        .groupby("source_role")
        .head(3)
        .copy()
    )

    top_transitions.to_csv(TRANSITION_SUMMARY_PATH, index=False)

    print("\nML dashboard summary files created successfully.")
    print("Saved:", CLASS_SUMMARY_PATH)
    print("Saved:", EMERGING_SUMMARY_PATH)
    print("Saved:", ANOMALY_SUMMARY_PATH)
    print("Saved:", TRANSITION_SUMMARY_PATH)

    print("\nShapes:")
    print("classification_summary:", classification_summary.shape)
    print("emerging_summary:", emerging_summary.shape)
    print("anomaly_summary:", anomaly_summary.shape)
    print("transition_summary:", top_transitions.shape)

    print("\nPreview - classification summary:")
    print(classification_summary.head(10))

    print("\nPreview - top transitions:")
    print(top_transitions.head(10))


if __name__ == "__main__":
    main()