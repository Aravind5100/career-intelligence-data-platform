import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path("data/ml_ready/feature_skill_weekly.csv")
OUTPUT_DIR = Path("outputs/ml")
EXPORT_DIR = Path("data/exports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PATH = OUTPUT_DIR / "forecast_model_comparison.csv"
FORECAST_EXPORT_PATH = EXPORT_DIR / "ml_skill_forecasts.csv"

TEST_WEEKS = 12


def naive_forecast(train_series, steps):
    last_value = train_series.iloc[-1]
    return np.repeat(last_value, steps)


def exp_smoothing_forecast(train_series, steps):
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal=None
    )
    fit = model.fit(optimized=True)
    return fit.forecast(steps)


def arima_forecast(train_series, steps):
    model = ARIMA(train_series, order=(1, 1, 1))
    fit = model.fit()
    return fit.forecast(steps)


def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = root_mean_squared_error(actual, forecast)
    return mae, rmse


def main():
    print("Reading weekly skill feature table...")
    df = pd.read_csv(INPUT_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])

    all_results = []
    all_forecasts = []

    skills = sorted(df["skill"].unique())
    print("Skills to forecast:")
    print(skills)

    for skill in skills:
        print(f"\nProcessing skill: {skill}")

        skill_df = df[df["skill"] == skill].sort_values("week_start").copy()

        series = skill_df["demand_count"].reset_index(drop=True)
        weeks = skill_df["week_start"].reset_index(drop=True)

        if len(series) <= TEST_WEEKS:
            print(f"Skipping {skill}: not enough observations.")
            continue

        train = series.iloc[:-TEST_WEEKS]
        test = series.iloc[-TEST_WEEKS:]
        test_weeks = weeks.iloc[-TEST_WEEKS:]

        # -----------------------------
        # 1. Naive baseline
        # -----------------------------
        naive_preds = naive_forecast(train, TEST_WEEKS)
        naive_mae, naive_rmse = evaluate_forecast(test, naive_preds)

        all_results.append({
            "skill": skill,
            "model_name": "naive",
            "mae": round(naive_mae, 4),
            "rmse": round(naive_rmse, 4)
        })

        for wk, actual, pred in zip(test_weeks, test, naive_preds):
            all_forecasts.append({
                "skill": skill,
                "week_start": wk,
                "actual_count": actual,
                "forecast_count": pred,
                "model_name": "naive"
            })

        # -----------------------------
        # 2. Exponential Smoothing
        # -----------------------------
        try:
            exp_preds = exp_smoothing_forecast(train, TEST_WEEKS)
            exp_mae, exp_rmse = evaluate_forecast(test, exp_preds)

            all_results.append({
                "skill": skill,
                "model_name": "exp_smoothing",
                "mae": round(exp_mae, 4),
                "rmse": round(exp_rmse, 4)
            })

            for wk, actual, pred in zip(test_weeks, test, exp_preds):
                all_forecasts.append({
                    "skill": skill,
                    "week_start": wk,
                    "actual_count": actual,
                    "forecast_count": pred,
                    "model_name": "exp_smoothing"
                })

        except Exception as e:
            print(f"Exponential Smoothing failed for {skill}: {e}")

        # -----------------------------
        # 3. ARIMA
        # -----------------------------
        try:
            arima_preds = arima_forecast(train, TEST_WEEKS)
            arima_mae, arima_rmse = evaluate_forecast(test, arima_preds)

            all_results.append({
                "skill": skill,
                "model_name": "arima",
                "mae": round(arima_mae, 4),
                "rmse": round(arima_rmse, 4)
            })

            for wk, actual, pred in zip(test_weeks, test, arima_preds):
                all_forecasts.append({
                    "skill": skill,
                    "week_start": wk,
                    "actual_count": actual,
                    "forecast_count": pred,
                    "model_name": "arima"
                })

        except Exception as e:
            print(f"ARIMA failed for {skill}: {e}")

    results_df = pd.DataFrame(all_results)
    forecasts_df = pd.DataFrame(all_forecasts)

    results_df.to_csv(COMPARISON_PATH, index=False)
    forecasts_df.to_csv(FORECAST_EXPORT_PATH, index=False)

    print("\nForecast model comparison saved to:", COMPARISON_PATH)
    print("Forecast output saved to:", FORECAST_EXPORT_PATH)

    print("\nComparison preview:")
    print(results_df.sort_values(["skill", "rmse"]).head(20))

    print("\nForecast output shape:", forecasts_df.shape)


if __name__ == "__main__":
    main()