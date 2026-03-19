import pandas as pd
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
JOBS_PATH = Path("data/processed/jobs_clean.parquet")
SKILLS_PATH = Path("data/processed/job_skills.parquet")

EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Reading Silver datasets...")
    jobs = pd.read_parquet(JOBS_PATH)
    skills = pd.read_parquet(SKILLS_PATH)

    # -----------------------------
    # Basic preparation
    # -----------------------------
    jobs["job_posted_date"] = pd.to_datetime(jobs["job_posted_date"], errors="coerce")

    # Work mode label
    jobs["work_mode"] = jobs["job_work_from_home"].map({
        True: "Remote",
        False: "Not Remote"
    })

    # Clean location label
    jobs["job_location_clean"] = jobs["job_location"].replace({
        "Anywhere": "Remote / Anywhere"
    })

    # --------------------------------------------------------
    # GOLD 1: Role Summary
    # --------------------------------------------------------
    print("Building gold_role_summary...")
    gold_role_summary = (
        jobs.groupby("job_title_short")
        .agg(
            total_jobs=("job_key", "count"),
            remote_jobs=("job_work_from_home", lambda x: (x == True).sum()),
            avg_salary_year=("salary_year_avg", "mean"),
            median_salary_year=("salary_year_avg", "median"),
            avg_salary_hour=("salary_hour_avg", "mean"),
            unique_companies=("company_name", "nunique"),
            unique_locations=("job_location_clean", "nunique")
        )
        .reset_index()
    )

    gold_role_summary["remote_pct"] = (
        gold_role_summary["remote_jobs"] / gold_role_summary["total_jobs"]
    ).round(4)

    # --------------------------------------------------------
    # GOLD 2: Location Summary
    # --------------------------------------------------------
    print("Building gold_location_summary...")
    gold_location_summary = (
        jobs.groupby("job_location_clean")
        .agg(
            total_jobs=("job_key", "count"),
            remote_jobs=("job_work_from_home", lambda x: (x == True).sum()),
            unique_roles=("job_title_short", "nunique"),
            unique_companies=("company_name", "nunique")
        )
        .reset_index()
    )

    gold_location_summary["remote_pct"] = (
        gold_location_summary["remote_jobs"] / gold_location_summary["total_jobs"]
    ).round(4)

    # --------------------------------------------------------
    # GOLD 3: Skill Summary
    # --------------------------------------------------------
    print("Building gold_skill_summary...")
    skill_with_jobs = skills.merge(
        jobs[["job_key", "job_title_short", "company_name"]],
        on="job_key",
        how="inner"
    )

    gold_skill_summary = (
        skill_with_jobs.groupby("skill")
        .agg(
            total_demand=("job_key", "count"),
            unique_roles=("job_title_short", "nunique"),
            unique_companies=("company_name", "nunique")
        )
        .reset_index()
        .sort_values("total_demand", ascending=False)
    )

    # --------------------------------------------------------
    # GOLD 4: Role-Skill Matrix
    # --------------------------------------------------------
    print("Building gold_role_skill_matrix...")
    gold_role_skill_matrix = (
        skill_with_jobs.groupby(["job_title_short", "skill"])
        .size()
        .reset_index(name="demand_count")
        .sort_values(["job_title_short", "demand_count"], ascending=[True, False])
    )

    # --------------------------------------------------------
    # GOLD 5: Weekly Job Trend
    # --------------------------------------------------------
    print("Building gold_jobs_weekly...")
    gold_jobs_weekly = (
        jobs.set_index("job_posted_date")
        .resample("W")
        .size()
        .reset_index(name="job_count")
    )
    gold_jobs_weekly = gold_jobs_weekly.rename(columns={"job_posted_date": "week_start"})

    # --------------------------------------------------------
    # GOLD 6: Role vs Remote Summary
    # --------------------------------------------------------
    print("Building gold_role_remote_summary...")
    role_remote_counts = (
        jobs.groupby(["job_title_short", "work_mode"])
        .size()
        .reset_index(name="job_count")
    )

    role_remote_counts["percentage"] = (
        role_remote_counts.groupby("job_title_short")["job_count"]
        .transform(lambda x: x / x.sum())
        .round(4)
    )

    gold_role_remote_summary = role_remote_counts.copy()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    print("Saving Gold files...")

    gold_role_summary.to_csv(EXPORT_DIR / "gold_role_summary.csv", index=False)
    gold_location_summary.to_csv(EXPORT_DIR / "gold_location_summary.csv", index=False)
    gold_skill_summary.to_csv(EXPORT_DIR / "gold_skill_summary.csv", index=False)
    gold_role_skill_matrix.to_csv(EXPORT_DIR / "gold_role_skill_matrix.csv", index=False)
    gold_jobs_weekly.to_csv(EXPORT_DIR / "gold_jobs_weekly.csv", index=False)
    gold_role_remote_summary.to_csv(EXPORT_DIR / "gold_role_remote_summary.csv", index=False)

    print("Gold files created successfully.")
    print("\nOutput shapes:")
    print("gold_role_summary:", gold_role_summary.shape)
    print("gold_location_summary:", gold_location_summary.shape)
    print("gold_skill_summary:", gold_skill_summary.shape)
    print("gold_role_skill_matrix:", gold_role_skill_matrix.shape)
    print("gold_jobs_weekly:", gold_jobs_weekly.shape)
    print("gold_role_remote_summary:", gold_role_remote_summary.shape)


if __name__ == "__main__":
    main()