import boto3
import argparse
from pathlib import Path

# -----------------------------
# S3 Configuration
# -----------------------------
BUCKET_NAME = "career-intelligence-data-platform"

LAYER_CONFIG = {
    "bronze": {
        "local_dir": Path("data/raw"),
        "s3_prefix": "bronze/"
    },
    "silver": {
        "local_dir": Path("data/processed"),
        "s3_prefix": "silver/",
        "exclude_dirs": ["gold"]
    },
    "gold": {
        "local_dir": Path("data/exports"),
        "s3_prefix": "gold/",
        "include_prefix": "gold_"
    },
    "ml": {
        "local_dir": Path("data/exports"),
        "s3_prefix": "ml/",
        "include_prefix": "ml_",
        "extras": [
            {
                "local_dir": Path("data/exports"),
                "s3_prefix": "ml/summaries/",
                "include_prefix": "ml_summary_"
            }
        ]
    }
}


def upload_files(s3_client, bucket, local_dir, s3_prefix,
                 include_prefix=None, exclude_dirs=None):
    """Upload files from a local directory to an S3 prefix."""
    if not local_dir.exists():
        print(f"Directory not found: {local_dir}")
        return 0

    uploaded = 0

    for file_path in sorted(local_dir.iterdir()):
        if file_path.is_dir():
            if exclude_dirs and file_path.name in exclude_dirs:
                continue
            continue

        if include_prefix and not file_path.name.startswith(include_prefix):
            continue

        s3_key = f"{s3_prefix}{file_path.name}"

        print(f"  Uploading {file_path.name} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)
        uploaded += 1

    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Upload pipeline outputs to S3 by layer."
    )
    parser.add_argument(
        "--layer",
        required=True,
        choices=["bronze", "silver", "gold", "ml", "all"],
        help="Pipeline layer to upload: bronze, silver, gold, ml, or all"
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3")

    if args.layer == "all":
        layers = ["bronze", "silver", "gold", "ml"]
    else:
        layers = [args.layer]

    total_uploaded = 0

    for layer in layers:
        config = LAYER_CONFIG[layer]
        print(f"\n--- Uploading {layer.upper()} layer ---")

        count = upload_files(
            s3_client=s3_client,
            bucket=BUCKET_NAME,
            local_dir=config["local_dir"],
            s3_prefix=config["s3_prefix"],
            include_prefix=config.get("include_prefix"),
            exclude_dirs=config.get("exclude_dirs")
        )
        total_uploaded += count

        # Handle extras (e.g., ml/summaries/)
        for extra in config.get("extras", []):
            count = upload_files(
                s3_client=s3_client,
                bucket=BUCKET_NAME,
                local_dir=extra["local_dir"],
                s3_prefix=extra["s3_prefix"],
                include_prefix=extra.get("include_prefix")
            )
            total_uploaded += count

    print(f"\nUpload complete. Total files uploaded: {total_uploaded}")


if __name__ == "__main__":
    main()
