import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
JOBS_PATH = Path("data/processed/jobs_clean.parquet")
SKILLS_PATH = Path("data/processed/job_skills.parquet")
OUTPUT_DIR = Path("data/ml_ready")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "feature_skill_weekly.csv"


def main():
    print("Reading Silver datasets...")
    jobs = pd.read_parquet(JOBS_PATH)
    skills = pd.read_parquet(SKILLS_PATH)

    # -----------------------------
    # Prepare date field
    # -----------------------------
    jobs["job_posted_date"] = pd.to_datetime(jobs["job_posted_date"], errors="coerce")
    jobs = jobs.dropna(subset=["job_posted_date"])

    # -----------------------------
    # Select top 10 skills overall
    # -----------------------------
    top_skills = skills["skill"].value_counts().head(10).index.tolist()
    print("Top 10 skills selected for forecasting:")
    print(top_skills)

    skills_top = skills[skills["skill"].isin(top_skills)].copy()

    # -----------------------------
    # Merge skill table with posted date
    # -----------------------------
    merged = skills_top.merge(
        jobs[["job_key", "job_posted_date"]],
        on="job_key",
        how="inner"
    )

    # -----------------------------
    # Create weekly aggregation
    # -----------------------------
    merged["week_start"] = merged["job_posted_date"].dt.to_period("W").dt.start_time

    skill_weekly = (
        merged.groupby(["week_start", "skill"])
        .size()
        .reset_index(name="demand_count")
        .sort_values(["skill", "week_start"])
    )

    # -----------------------------
    # Save output
    # -----------------------------
    skill_weekly.to_csv(OUTPUT_PATH, index=False)

    print("\nWeekly skill feature table created successfully.")
    print("Output path:", OUTPUT_PATH)
    print("Shape:", skill_weekly.shape)

    print("\nSample rows:")
    print(skill_weekly.head(15))

    print("\nWeekly range:")
    print("Min week:", skill_weekly["week_start"].min())
    print("Max week:", skill_weekly["week_start"].max())


if __name__ == "__main__":
    main()