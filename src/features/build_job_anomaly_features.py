import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
JOBS_PATH = Path("data/processed/jobs_clean.parquet")
SKILLS_PATH = Path("data/processed/job_skills.parquet")
ROLE_PRED_PATH = Path("data/exports/ml_role_predictions.csv")

OUTPUT_DIR = Path("data/ml_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "feature_job_anomalies.csv"


def main():
    print("Reading inputs...")
    jobs = pd.read_parquet(JOBS_PATH)
    skills = pd.read_parquet(SKILLS_PATH)
    preds = pd.read_csv(ROLE_PRED_PATH)

    # -----------------------------
    # 1. Base jobs table
    # -----------------------------
    jobs["salary_year_avg"] = pd.to_numeric(jobs["salary_year_avg"], errors="coerce")

    base = jobs[[
        "job_key",
        "job_title",
        "company_name",
        "job_location",
        "job_title_short",
        "salary_year_avg"
    ]].copy()

    base = base.rename(columns={"job_title_short": "actual_role"})

    # -----------------------------
    # 2. Merge predictions
    # -----------------------------
    base = base.merge(
        preds[["job_key", "predicted_role", "correct_prediction"]],
        on="job_key",
        how="left"
    )

    base["role_disagreement_flag"] = (base["actual_role"] != base["predicted_role"]).astype(int)

    # -----------------------------
    # 3. Salary context by role
    # -----------------------------
    role_salary_stats = (
        base.groupby("actual_role")["salary_year_avg"]
        .median()
        .reset_index()
        .rename(columns={"salary_year_avg": "role_median_salary_year"})
    )

    base = base.merge(role_salary_stats, on="actual_role", how="left")

    base["salary_deviation_from_role_median"] = (
        base["salary_year_avg"] - base["role_median_salary_year"]
    )

    # Optional normalized ratio-style feature
    base["salary_ratio_to_role_median"] = (
        base["salary_year_avg"] / base["role_median_salary_year"]
    )

    # -----------------------------
    # 4. Skill count per job
    # -----------------------------
    job_skill_counts = (
        skills.groupby("job_key")
        .size()
        .reset_index(name="job_skill_count")
    )

    base = base.merge(job_skill_counts, on="job_key", how="left")
    base["job_skill_count"] = base["job_skill_count"].fillna(0)

    # -----------------------------
    # 5. Skill count context by role
    # -----------------------------
    role_skill_stats = (
        base.groupby("actual_role")["job_skill_count"]
        .median()
        .reset_index()
        .rename(columns={"job_skill_count": "role_median_skill_count"})
    )

    base = base.merge(role_skill_stats, on="actual_role", how="left")

    base["skill_count_deviation_from_role_median"] = (
        base["job_skill_count"] - base["role_median_skill_count"]
    )

    # Optional ratio feature
    base["skill_count_ratio_to_role_median"] = (
        base["job_skill_count"] / base["role_median_skill_count"]
    )

    # -----------------------------
    # 6. Simple rule flags
    # -----------------------------
    # Salary anomaly flag:
    # mark if salary is more than 50% above or below role median
    base["salary_anomaly_flag"] = (
        (base["salary_ratio_to_role_median"] > 1.5) |
        (base["salary_ratio_to_role_median"] < 0.5)
    ).astype(int)

    # Skill count anomaly flag:
    # mark if skill count is far above/below role median
    base["skill_count_anomaly_flag"] = (
        (base["skill_count_ratio_to_role_median"] > 1.5) |
        (base["skill_count_ratio_to_role_median"] < 0.5)
    ).astype(int)

    # -----------------------------
    # 7. Final feature table
    # -----------------------------
    anomaly_features = base[[
        "job_key",
        "job_title",
        "company_name",
        "job_location",
        "actual_role",
        "predicted_role",
        "correct_prediction",
        "role_disagreement_flag",
        "salary_year_avg",
        "role_median_salary_year",
        "salary_deviation_from_role_median",
        "salary_ratio_to_role_median",
        "salary_anomaly_flag",
        "job_skill_count",
        "role_median_skill_count",
        "skill_count_deviation_from_role_median",
        "skill_count_ratio_to_role_median",
        "skill_count_anomaly_flag"
    ]].copy()

    anomaly_features.to_csv(OUTPUT_PATH, index=False)

    print("\nAnomaly feature table created successfully.")
    print("Output path:", OUTPUT_PATH)
    print("Shape:", anomaly_features.shape)

    print("\nFlag summary:")
    print("Role disagreement count:", anomaly_features["role_disagreement_flag"].sum())
    print("Salary anomaly count:", anomaly_features["salary_anomaly_flag"].sum())
    print("Skill count anomaly count:", anomaly_features["skill_count_anomaly_flag"].sum())

    print("\nSample rows:")
    print(anomaly_features.head(10))


if __name__ == "__main__":
    main()