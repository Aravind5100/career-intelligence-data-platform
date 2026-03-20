import pandas as pd
from pathlib import Path
import ast

# -----------------------------
# Paths
# -----------------------------
SIMILARITY_PATH = Path("data/exports/ml_role_similarity.csv")
ROLE_VECTOR_PATH = Path("data/ml_ready/feature_role_skill_vectors.csv")
SKILL_SUMMARY_PATH = Path("data/exports/gold_skill_summary.csv")
EMERGING_PATH = Path("data/exports/ml_emerging_skills.csv")
EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = EXPORT_DIR / "ml_transition_recommendations.csv"

TOP_N_SKILLS = 15
TOP_N_RECOMMENDATIONS = 5


def normalize_series(s):
    if s.max() == s.min():
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def main():
    print("Reading inputs...")
    similarity = pd.read_csv(SIMILARITY_PATH)
    role_vectors = pd.read_csv(ROLE_VECTOR_PATH)
    skill_summary = pd.read_csv(SKILL_SUMMARY_PATH)
    emerging = pd.read_csv(EMERGING_PATH)

    # -----------------------------
    # Prepare supporting skill scores
    # -----------------------------
    skill_summary["normalized_market_demand"] = normalize_series(skill_summary["total_demand"])

    emerging["normalized_emerging_score"] = normalize_series(emerging["emerging_score"])

    skill_support = skill_summary.merge(
        emerging[["skill", "emerging_score", "normalized_emerging_score", "signal_label"]],
        on="skill",
        how="left"
    )

    skill_support["normalized_emerging_score"] = skill_support["normalized_emerging_score"].fillna(0)
    skill_support["emerging_score"] = skill_support["emerging_score"].fillna(0)
    skill_support["signal_label"] = skill_support["signal_label"].fillna("Stable")

    # -----------------------------
    # Keep only top 15 skills per role
    # -----------------------------
    top_role_skills = (
        role_vectors.sort_values(["job_title_short", "role_skill_weight"], ascending=[True, False])
        .groupby("job_title_short")
        .head(TOP_N_SKILLS)
        .copy()
    )

    recommendations = []

    # -----------------------------
    # Generate transition recommendations
    # -----------------------------
    for _, row in similarity.iterrows():
        source_role = row["source_role"]
        target_role = row["target_role"]
        similarity_score = row["similarity_score"]

        source_skills_df = top_role_skills[top_role_skills["job_title_short"] == source_role].copy()
        target_skills_df = top_role_skills[top_role_skills["job_title_short"] == target_role].copy()

        source_skills = set(source_skills_df["skill"])
        target_skills = set(target_skills_df["skill"])

        shared_skills = sorted(source_skills.intersection(target_skills))
        missing_skills = sorted(target_skills.difference(source_skills))

        if len(missing_skills) == 0:
            recommended_next_skills = []
        else:
            missing_skill_scores = target_skills_df[target_skills_df["skill"].isin(missing_skills)].copy()

            missing_skill_scores = missing_skill_scores.merge(
                skill_support[[
                    "skill",
                    "normalized_market_demand",
                    "normalized_emerging_score",
                    "signal_label"
                ]],
                on="skill",
                how="left"
            )

            missing_skill_scores["normalized_market_demand"] = missing_skill_scores["normalized_market_demand"].fillna(0)
            missing_skill_scores["normalized_emerging_score"] = missing_skill_scores["normalized_emerging_score"].fillna(0)

            missing_skill_scores["recommendation_score"] = (
                0.5 * missing_skill_scores["role_skill_weight"] +
                0.3 * missing_skill_scores["normalized_market_demand"] +
                0.2 * missing_skill_scores["normalized_emerging_score"]
            )

            missing_skill_scores = missing_skill_scores.sort_values(
                "recommendation_score", ascending=False
            )

            recommended_next_skills = missing_skill_scores["skill"].head(TOP_N_RECOMMENDATIONS).tolist()

        recommendations.append({
            "source_role": source_role,
            "target_role": target_role,
            "similarity_score": round(similarity_score, 4),
            "shared_skills": ", ".join(shared_skills),
            "missing_skills": ", ".join(missing_skills),
            "recommended_next_skills": ", ".join(recommended_next_skills)
        })

    recommendations_df = pd.DataFrame(recommendations)

    # Sort so the strongest transitions appear first within each source role
    recommendations_df = recommendations_df.sort_values(
        ["source_role", "similarity_score"],
        ascending=[True, False]
    )

    recommendations_df.to_csv(OUTPUT_PATH, index=False)

    print("\nTransition recommendation table created successfully.")
    print("Output path:", OUTPUT_PATH)
    print("Shape:", recommendations_df.shape)

    print("\nTop transition recommendations:")
    print(recommendations_df.head(20))


if __name__ == "__main__":
    main()