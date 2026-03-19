import pandas as pd
from pathlib import Path
import re
import hashlib

# -----------------------------
# File paths
# -----------------------------
RAW_PATH = Path("data/raw/jobs_raw.parquet")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

JOBS_OUTPUT = PROCESSED_DIR / "jobs_clean.parquet"
SKILLS_OUTPUT = PROCESSED_DIR / "job_skills.parquet"


# -----------------------------
# Helper functions
# -----------------------------
def normalize_text(value):
    """
    Clean text fields by:
    - handling nulls
    - trimming extra spaces
    - reducing repeated whitespace
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    return value if value else None


def make_job_key(row):
    """
    Create a hashed key from important identifying fields.
    This helps create a compact, consistent unique identifier.
    """
    key_string = (
        f"{row.get('job_title', '')}|"
        f"{row.get('company_name', '')}|"
        f"{row.get('job_location', '')}|"
        f"{row.get('job_posted_date', '')}"
    )
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()


def parse_skill_list(value):
    """
    Convert job_skills into a Python list.
    Handles:
    - nulls
    - already-list values
    - string-like list values
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [normalize_text(v) for v in value if normalize_text(v)]

    value = str(value).strip()

    # Remove brackets if stored like: [python, sql]
    value = value.strip("[]")

    # Split by comma
    parts = [p.strip().strip("'").strip('"') for p in value.split(",")]

    # Clean each part
    cleaned = [normalize_text(p) for p in parts if normalize_text(p)]

    return cleaned


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    print("Reading Bronze data...")
    df = pd.read_parquet(RAW_PATH)

    print("Initial shape:", df.shape)

    # -------------------------
    # 1. Normalize text columns
    # -------------------------
    text_cols = [
        "job_title_short",
        "job_title",
        "job_location",
        "job_via",
        "job_schedule_type",
        "search_location",
        "job_country",
        "salary_rate",
        "company_name",
        "job_type_skills"
    ]

    for col in text_cols:
        df[col] = df[col].apply(normalize_text)

    # -------------------------
    # 2. Parse dates
    # -------------------------
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], errors="coerce")

    # -------------------------
    # 3. Convert salary columns to numeric
    # -------------------------
    df["salary_year_avg"] = pd.to_numeric(df["salary_year_avg"], errors="coerce")
    df["salary_hour_avg"] = pd.to_numeric(df["salary_hour_avg"], errors="coerce")

    # -------------------------
    # 4. Normalize boolean columns
    # -------------------------
    bool_cols = [
        "job_work_from_home",
        "job_no_degree_mention",
        "job_health_insurance"
    ]

    for col in bool_cols:
        df[col] = df[col].astype("boolean")

    # -------------------------
    # 5. Remove duplicates
    # -------------------------
    before = len(df)

    df = df.drop_duplicates(
        subset=["job_title", "company_name", "job_location", "job_posted_date"]
    )

    after = len(df)
    removed_duplicates = before - after

    print(f"Duplicates removed: {removed_duplicates}")

    # -------------------------
    # 6. Generate hashed job key
    # -------------------------
    df["job_key"] = df.apply(make_job_key, axis=1)

    # -------------------------
    # 7. Parse and explode job_skills
    # -------------------------
    df["skills_list"] = df["job_skills"].apply(parse_skill_list)

    job_skills = df[["job_key", "skills_list"]].explode("skills_list")
    job_skills = job_skills.rename(columns={"skills_list": "skill"})

    # Clean skill column
    job_skills["skill"] = job_skills["skill"].apply(normalize_text)

    # Remove empty/null skill rows
    job_skills = job_skills.dropna(subset=["skill"])
    job_skills = job_skills[job_skills["skill"] != ""]

    # Remove duplicate job-skill pairs
    job_skills = job_skills.drop_duplicates(subset=["job_key", "skill"])

    # -------------------------
    # 8. Drop temporary columns and save outputs
    # -------------------------
    jobs_clean = df.drop(columns=["skills_list"])

    jobs_clean.to_parquet(JOBS_OUTPUT, index=False)
    job_skills.to_parquet(SKILLS_OUTPUT, index=False)

    print("Saved clean jobs file:", JOBS_OUTPUT)
    print("Saved job skills file:", SKILLS_OUTPUT)
    print("jobs_clean shape:", jobs_clean.shape)
    print("job_skills shape:", job_skills.shape)


if __name__ == "__main__":
    main()