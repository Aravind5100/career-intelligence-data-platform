import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
FORECAST_PATH = Path("data/exports/ml_skill_forecasts.csv")
EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = EXPORT_DIR / "ml_emerging_skills.csv"


def pct_change_safe(new_value, old_value):
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def label_signal(score):
    if score > 2:
        return "Emerging"
    elif score < -2:
        return "Cooling"
    return "Stable"


def main():
    print("Reading forecast output...")
    df = pd.read_csv(FORECAST_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])

    # We only want the best-performing forecast model per skill.
    # From your results, Exponential Smoothing appears strongest overall,
    # so we'll use it as the signal model.
    forecast_df = df[df["model_name"] == "exp_smoothing"].copy()

    results = []

    for skill in sorted(forecast_df["skill"].unique()):
        skill_df = forecast_df[forecast_df["skill"] == skill].sort_values("week_start").copy()

        # Test horizon contains actual + forecast values for the last 12 weeks
        # We use the last 4 actual points and the first 4 forecasted points as comparison windows
        last_4_actual = skill_df["actual_count"].tail(4).mean()
        prior_4_actual = skill_df["actual_count"].tail(8).head(4).mean()
        avg_forecast = skill_df["forecast_count"].head(4).mean()

        recent_growth_pct = pct_change_safe(last_4_actual, prior_4_actual)
        forecast_growth_pct = pct_change_safe(avg_forecast, last_4_actual)

        emerging_score = round((0.5 * recent_growth_pct) + (0.5 * forecast_growth_pct), 2)
        signal_label = label_signal(emerging_score)

        results.append({
            "skill": skill,
            "prior_4wk_actual_avg": round(prior_4_actual, 2),
            "last_4wk_actual_avg": round(last_4_actual, 2),
            "next_4wk_forecast_avg": round(avg_forecast, 2),
            "recent_growth_pct": round(recent_growth_pct, 2),
            "forecast_growth_pct": round(forecast_growth_pct, 2),
            "emerging_score": emerging_score,
            "signal_label": signal_label
        })

    emerging_df = pd.DataFrame(results).sort_values("emerging_score", ascending=False)
    emerging_df.to_csv(OUTPUT_PATH, index=False)

    print("\nEmerging skill signal table created successfully.")
    print("Output path:", OUTPUT_PATH)
    print("Shape:", emerging_df.shape)

    print("\nTop emerging skills:")
    print(emerging_df.head(10))

    print("\nSignal label distribution:")
    print(emerging_df["signal_label"].value_counts())


if __name__ == "__main__":
    main()