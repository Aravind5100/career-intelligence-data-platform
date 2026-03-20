import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/exports/gold_role_skill_matrix.csv")
ML_READY_DIR = Path("data/ml_ready")
EXPORT_DIR = Path("data/exports")

ML_READY_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_VECTOR_PATH = ML_READY_DIR / "feature_role_skill_vectors.csv"
SIMILARITY_PATH = EXPORT_DIR / "ml_role_similarity.csv"


def main():
    print("Reading gold role-skill matrix...")
    df = pd.read_csv(INPUT_PATH)

    # -----------------------------------
    # 1. Normalize demand_count within role
    # -----------------------------------
    df["role_skill_weight"] = (
        df["demand_count"] /
        df.groupby("job_title_short")["demand_count"].transform("sum")
    )

    # Save long-form role-skill weighted table
    role_vectors_long = df[[
        "job_title_short",
        "skill",
        "demand_count",
        "role_skill_weight"
    ]].copy()

    role_vectors_long.to_csv(ROLE_VECTOR_PATH, index=False)

    # -----------------------------------
    # 2. Pivot into wide role-skill matrix
    # -----------------------------------
    role_skill_matrix = role_vectors_long.pivot_table(
        index="job_title_short",
        columns="skill",
        values="role_skill_weight",
        fill_value=0
    )

    # -----------------------------------
    # 3. Compute cosine similarity
    # -----------------------------------
    similarity_matrix = cosine_similarity(role_skill_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=role_skill_matrix.index,
        columns=role_skill_matrix.index
    )

    # -----------------------------------
    # 4. Convert to long table
    # -----------------------------------
    similarity_long = (
        similarity_df.reset_index()
        .melt(id_vars="job_title_short", var_name="target_role", value_name="similarity_score")
        .rename(columns={"job_title_short": "source_role"})
    )

    # Remove self-comparisons
    similarity_long = similarity_long[
        similarity_long["source_role"] != similarity_long["target_role"]
    ].copy()

    similarity_long = similarity_long.sort_values(
        ["source_role", "similarity_score"],
        ascending=[True, False]
    )

    similarity_long.to_csv(SIMILARITY_PATH, index=False)

    print("\nRole similarity table created successfully.")
    print("Role vector path:", ROLE_VECTOR_PATH)
    print("Similarity path:", SIMILARITY_PATH)

    print("\nTop similarity pairs:")
    print(similarity_long.head(20))

    print("\nRole-skill vector shape:", role_skill_matrix.shape)
    print("Similarity table shape:", similarity_long.shape)


if __name__ == "__main__":
    main()