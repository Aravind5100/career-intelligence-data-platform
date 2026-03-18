from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Define output folder
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading Dataset from Hugging Face...")
    ds = load_dataset("lukebarousse/data_jobs", split="train")

    print("Converting to pandas DataFrame...")
    df = ds.to_pandas()

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    output_path = RAW_DIR / "jobs_raw.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Raw dataset saved to: {output_path}")

if __name__ == "__main__":
    main()