import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------
# File paths
# -----------------------------
RAW_PATH = Path("data/raw/jobs_raw.parquet")
JOBS_PATH = Path("data/processed/jobs_clean.parquet")
SKILLS_PATH = Path("data/processed/job_skills.parquet")
METADATA_PATH = Path("data/processed/pipeline_run_metadata.csv")


def main():
    print("Reading pipeline files...")

    raw = pd.read_parquet(RAW_PATH)
    jobs = pd.read_parquet(JOBS_PATH)
    skills = pd.read_parquet(SKILLS_PATH)

    input_rows = len(raw)
    output_jobs_rows = len(jobs)
    output_skills_rows = len(skills)
    duplicates_removed = input_rows - output_jobs_rows

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_record = pd.DataFrame([{
        "run_timestamp": run_timestamp,
        "input_file": str(RAW_PATH),
        "output_jobs_file": str(JOBS_PATH),
        "output_skills_file": str(SKILLS_PATH),
        "input_rows": input_rows,
        "output_jobs_rows": output_jobs_rows,
        "output_skills_rows": output_skills_rows,
        "duplicates_removed": duplicates_removed,
        "status": "success"
    }])

    if METADATA_PATH.exists():
        existing = pd.read_csv(METADATA_PATH)
        updated = pd.concat([existing, run_record], ignore_index=True)
    else:
        updated = run_record

    updated.to_csv(METADATA_PATH, index=False)

    print("Pipeline metadata recorded successfully.")
    print(updated.tail(5))
    print(f"Saved metadata file to: {METADATA_PATH}")


if __name__ == "__main__":
    main()