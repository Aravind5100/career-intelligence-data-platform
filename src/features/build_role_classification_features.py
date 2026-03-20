import pandas as pd
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
JOBS_PATH = Path("data/processed/jobs_clean.parquet")
OUTPUT_DIR = Path("data/ml_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "feature_role_classification.csv"


def main():
    print("Reading Silver jobs data...")
    jobs = pd.read_parquet(JOBS_PATH)

    # -----------------------------------
    # 1. Create cleaned location field
    # -----------------------------------
    jobs["job_location_clean"] = jobs["job_location"].replace({
        "Anywhere": "Remote / Anywhere"
    })

    # -----------------------------------
    # 2. Select top 8 roles by volume
    # -----------------------------------
    top_roles = jobs["job_title_short"].value_counts().head(8).index.tolist()
    print("Top 8 roles selected for classification:")
    print(top_roles)

    classification_df = jobs[jobs["job_title_short"].isin(top_roles)].copy()

    # -----------------------------------
    # 3. Build combined text feature
    # -----------------------------------
    classification_df["text_blob"] = (
        classification_df["job_title"].fillna("") + " " +
        classification_df["job_type_skills"].fillna("") + " " +
        classification_df["job_location_clean"].fillna("")
    ).str.strip()

    # -----------------------------------
    # 4. Keep only required columns
    # -----------------------------------
    classification_features = classification_df[[
        "job_key",
        "job_title_short",          # target label
        "text_blob",                # main text feature
        "job_work_from_home",
        "job_no_degree_mention",
        "job_health_insurance",
        "salary_year_avg",
        "salary_hour_avg",
        "salary_rate"
    ]].copy()

    # -----------------------------------
    # 5. Basic sanity cleanup
    # -----------------------------------
    # Remove rows where text_blob is empty
    classification_features = classification_features[
        classification_features["text_blob"].str.len() > 0
    ]

    # -----------------------------------
    # 6. Save output
    # -----------------------------------
    classification_features.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature table created successfully.")
    print("Output path:", OUTPUT_PATH)
    print("Shape:", classification_features.shape)

    print("\nClass distribution:")
    print(classification_features["job_title_short"].value_counts())

    print("\nMissing values summary:")
    print(classification_features.isna().sum())


if __name__ == "__main__":
    main()