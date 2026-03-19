import pandas as pd
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
JOBS_PATH = Path("data/processed/jobs_clean.parquet")
SKILLS_PATH = Path("data/processed/job_skills.parquet")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "quality_report.txt"


def main():
    print("Reading Silver files...")
    jobs = pd.read_parquet(JOBS_PATH)
    skills = pd.read_parquet(SKILLS_PATH)

    # -----------------------------
    # 1. Basic row counts
    # -----------------------------
    jobs_count = len(jobs)
    skills_count = len(skills)

    # -----------------------------
    # 2. Null checks for critical columns
    # -----------------------------
    critical_cols = [
        "job_title",
        "company_name",
        "job_location",
        "job_posted_date",
        "job_title_short"
    ]

    null_counts = jobs[critical_cols].isna().sum()
    null_percentages = (null_counts / jobs_count * 100).round(2)

    # -----------------------------
    # 3. Duplicate checks
    # -----------------------------
    duplicate_job_keys = jobs["job_key"].duplicated().sum()
    duplicate_job_skill_pairs = skills.duplicated(subset=["job_key", "skill"]).sum()

    # -----------------------------
    # 4. Salary sanity checks
    # -----------------------------
    salary_year_summary = jobs["salary_year_avg"].describe()
    salary_hour_summary = jobs["salary_hour_avg"].describe()

    # Check for negative salaries (should usually be zero)
    negative_salary_year = (jobs["salary_year_avg"] < 0).sum()
    negative_salary_hour = (jobs["salary_hour_avg"] < 0).sum()

    # -----------------------------
    # 5. Boolean field sanity
    # -----------------------------
    bool_cols = [
        "job_work_from_home",
        "job_no_degree_mention",
        "job_health_insurance"
    ]

    bool_value_counts = {}
    for col in bool_cols:
        bool_value_counts[col] = jobs[col].value_counts(dropna=False).to_dict()

    # -----------------------------
    # 6. Skill table quality
    # -----------------------------
    null_skills = skills["skill"].isna().sum()
    empty_skills = (skills["skill"].astype(str).str.strip() == "").sum()
    unique_skills = skills["skill"].nunique()

    # -----------------------------
    # 7. Build report text
    # -----------------------------
    lines = []
    lines.append("QUALITY REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("1. ROW COUNTS")
    lines.append(f"Jobs rows: {jobs_count}")
    lines.append(f"Job-skills rows: {skills_count}")
    lines.append("")

    lines.append("2. CRITICAL COLUMN NULL CHECKS")
    for col in critical_cols:
        lines.append(
            f"{col}: {null_counts[col]} nulls ({null_percentages[col]}%)"
        )
    lines.append("")

    lines.append("3. DUPLICATE CHECKS")
    lines.append(f"Duplicate job_key values: {duplicate_job_keys}")
    lines.append(f"Duplicate job_key-skill pairs: {duplicate_job_skill_pairs}")
    lines.append("")

    lines.append("4. SALARY SANITY CHECKS")
    lines.append("Salary Year Avg Summary:")
    lines.append(str(salary_year_summary))
    lines.append("")
    lines.append("Salary Hour Avg Summary:")
    lines.append(str(salary_hour_summary))
    lines.append("")
    lines.append(f"Negative annual salary rows: {negative_salary_year}")
    lines.append(f"Negative hourly salary rows: {negative_salary_hour}")
    lines.append("")

    lines.append("5. BOOLEAN FIELD VALUE COUNTS")
    for col, values in bool_value_counts.items():
        lines.append(f"{col}: {values}")
    lines.append("")

    lines.append("6. SKILL TABLE QUALITY")
    lines.append(f"Null skills: {null_skills}")
    lines.append(f"Empty skills: {empty_skills}")
    lines.append(f"Unique skills: {unique_skills}")
    lines.append("")

    # -----------------------------
    # 8. Print report to console
    # -----------------------------
    report_text = "\n".join(lines)
    print(report_text)

    # -----------------------------
    # 9. Save report to file
    # -----------------------------
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print("")
    print(f"Quality report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()