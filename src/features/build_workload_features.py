import argparse
from pathlib import Path

import pandas as pd


def build_workload_features(silver_dir: str | Path, gold_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    silver_dir = Path(silver_dir)
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(silver_dir / "summary_hourly_silver.csv")
    if summary.empty:
        features = pd.DataFrame()
        targets = pd.DataFrame()
        features.to_csv(gold_dir / "workload_features_hourly.csv", index=False)
        targets.to_csv(gold_dir / "workload_targets_hourly.csv", index=False)
        return features, targets

    summary["available_at_hour"] = pd.to_datetime(summary["available_at_hour"], utc=True, errors="coerce")
    summary = summary.sort_values("available_at_hour").reset_index(drop=True)

    summary["hour_of_day"] = summary["available_at_hour"].dt.hour
    summary["day_of_week"] = summary["available_at_hour"].dt.dayofweek
    summary["rolling_mean_24h"] = summary["review_count"].rolling(window=24, min_periods=1).mean()
    summary["rolling_std_24h"] = summary["review_count"].rolling(window=24, min_periods=1).std().fillna(0)
    summary["reviews_due_next_1h"] = summary["review_count"].shift(-1).fillna(0)
    summary["reviews_due_next_3h"] = (
        summary["review_count"].shift(-1).fillna(0)
        + summary["review_count"].shift(-2).fillna(0)
        + summary["review_count"].shift(-3).fillna(0)
    )
    summary["reviews_due_next_24h"] = sum(summary["review_count"].shift(-i).fillna(0) for i in range(1, 25))

    features = summary[[
        "available_at_hour", "review_count", "lesson_count", "hour_of_day", "day_of_week",
        "rolling_mean_24h", "rolling_std_24h", "reviews_due_next_1h", "reviews_due_next_3h",
        "reviews_due_next_24h",
    ]].rename(columns={"available_at_hour": "hour_ts", "review_count": "reviews_due_now"})

    targets = pd.DataFrame({
        "hour_ts": features["hour_ts"],
        "target_reviews_next_hour": summary["review_count"].shift(-1).fillna(0),
        "target_reviews_next_24h": summary["reviews_due_next_24h"],
    })

    features.to_csv(gold_dir / "workload_features_hourly.csv", index=False)
    targets.to_csv(gold_dir / "workload_targets_hourly.csv", index=False)

    return features, targets


def main(silver_dir: str, gold_dir: str) -> None:
    features, targets = build_workload_features(silver_dir=silver_dir, gold_dir=gold_dir)
    print(f"Saved {len(features)} workload feature rows to {Path(gold_dir) / 'workload_features_hourly.csv'}")
    print(f"Saved {len(targets)} workload target rows to {Path(gold_dir) / 'workload_targets_hourly.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build workload forecasting features.")
    parser.add_argument("--silver-dir", default="data/silver", help="Directory containing silver CSV files.")
    parser.add_argument("--gold-dir", default="data/gold", help="Output directory for gold feature files.")
    args = parser.parse_args()

    main(silver_dir=args.silver_dir, gold_dir=args.gold_dir)
