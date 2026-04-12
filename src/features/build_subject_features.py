import argparse
from pathlib import Path

import pandas as pd


def build_subject_features(silver_dir: str | Path, gold_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    silver_dir = Path(silver_dir)
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(silver_dir / "assignments_silver.csv")
    review_stats = pd.read_csv(silver_dir / "review_stats_silver.csv")
    subjects = pd.read_csv(silver_dir / "subjects_silver.csv")

    merged = (
        assignments
        .merge(review_stats, on="subject_id", how="left", suffixes=("_assignment", "_review"))
        .merge(subjects, on="subject_id", how="left", suffixes=("", "_subject"))
    )

    for col in [
        "data_updated_at_assignment", "data_created_at_assignment", "data_available_at",
        "data_unlocked_at", "data_started_at", "data_passed_at", "data_burned_at",
        "data_updated_at_review", "data_created_at_review", "data_updated_at", "data_created_at",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], utc=True, errors="coerce")

    now_ts = pd.Timestamp.now(tz="UTC")

    merged["snapshot_date"] = now_ts.date().isoformat()
    merged["pct_correct"] = merged.get("data_percentage_correct", 0).fillna(0)
    merged["meaning_correct"] = merged.get("data_meaning_correct", 0).fillna(0)
    merged["reading_correct"] = merged.get("data_reading_correct", 0).fillna(0)
    merged["meaning_incorrect"] = merged.get("data_meaning_incorrect", 0).fillna(0)
    merged["reading_incorrect"] = merged.get("data_reading_incorrect", 0).fillna(0)

    merged["total_correct"] = merged["meaning_correct"] + merged["reading_correct"]
    merged["total_incorrect"] = merged["meaning_incorrect"] + merged["reading_incorrect"]
    merged["total_attempts_est"] = merged["total_correct"] + merged["total_incorrect"]

    merged["meaning_error_rate"] = (merged["meaning_incorrect"] / (merged["meaning_correct"] + merged["meaning_incorrect"]).replace(0, pd.NA)).fillna(0)
    merged["reading_error_rate"] = (merged["reading_incorrect"] / (merged["reading_correct"] + merged["reading_incorrect"]).replace(0, pd.NA)).fillna(0)

    if "data_unlocked_at" in merged.columns:
        merged["age_days_since_unlocked"] = (now_ts - merged["data_unlocked_at"]).dt.total_seconds().div(86400).clip(lower=0)
    else:
        merged["age_days_since_unlocked"] = 0

    if "data_updated_at_assignment" in merged.columns:
        merged["days_since_assignment_update"] = (now_ts - merged["data_updated_at_assignment"]).dt.total_seconds().div(86400).clip(lower=0)
    else:
        merged["days_since_assignment_update"] = 0

    merged["risk_label_binary"] = (
        (merged["pct_correct"] < 80)
        | ((merged["total_incorrect"] / merged["total_attempts_est"].replace(0, pd.NA)).fillna(0) > 0.20)
        | (merged.get("data_srs_stage", 9).fillna(9) <= 3)
    ).astype(int)

    merged["risk_label_multiclass"] = "low"
    merged.loc[(merged["pct_correct"] < 90) | (merged.get("data_srs_stage", 9).fillna(9).between(4, 5)), "risk_label_multiclass"] = "medium"
    merged.loc[(merged["pct_correct"] < 80) | (merged.get("data_srs_stage", 9).fillna(9) <= 3), "risk_label_multiclass"] = "high"

    feature_cols = [
        "snapshot_date", "subject_id", "subject_type", "data_level", "data_srs_stage", "pct_correct",
        "total_attempts_est", "total_incorrect", "meaning_error_rate", "reading_error_rate",
        "data_meaning_current_streak", "data_reading_current_streak", "data_meaning_max_streak",
        "data_reading_max_streak", "age_days_since_unlocked", "days_since_assignment_update", "data_hidden",
        "has_reading", "component_subject_count", "amalgamation_subject_count", "visually_similar_count",
        "parts_of_speech_count",
    ]
    feature_cols = [c for c in feature_cols if c in merged.columns]

    features = merged[feature_cols].copy()
    labels = merged[["snapshot_date", "subject_id", "risk_label_binary", "risk_label_multiclass"]].copy()

    features.to_csv(gold_dir / "subject_features_daily.csv", index=False)
    labels.to_csv(gold_dir / "risk_labels_daily.csv", index=False)

    return features, labels


def main(silver_dir: str, gold_dir: str) -> None:
    features, labels = build_subject_features(silver_dir=silver_dir, gold_dir=gold_dir)
    print(f"Saved {len(features)} subject feature rows to {Path(gold_dir) / 'subject_features_daily.csv'}")
    print(f"Saved {len(labels)} label rows to {Path(gold_dir) / 'risk_labels_daily.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build subject-level gold features and risk labels.")
    parser.add_argument("--silver-dir", default="data/silver", help="Directory containing silver CSV files.")
    parser.add_argument("--gold-dir", default="data/gold", help="Output directory for gold feature files.")
    args = parser.parse_args()

    main(silver_dir=args.silver_dir, gold_dir=args.gold_dir)
