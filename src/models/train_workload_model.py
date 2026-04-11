import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def train_workload_model(gold_dir: str | Path, model_dir: str | Path, random_state: int = 42) -> dict:
    gold_dir = Path(gold_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(gold_dir / "workload_features_hourly.csv")
    targets = pd.read_csv(gold_dir / "workload_targets_hourly.csv")

    data = features.merge(targets, on="hour_ts", how="inner")
    if data.empty:
        raise ValueError("No training rows found. Build workload gold tables first.")

    target_col = "target_reviews_next_hour"
    drop_cols = ["hour_ts", "target_reviews_next_hour", "target_reviews_next_24h"]
    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    artifact_path = model_dir / "workload_model.joblib"
    joblib.dump(model, artifact_path)

    metrics = {
        "mae_next_hour": float(mae),
        "artifact_path": str(artifact_path),
    }
    pd.DataFrame([metrics]).to_csv(model_dir / "workload_model_metrics.csv", index=False)
    return metrics


def main(gold_dir: str, model_dir: str, random_state: int) -> None:
    metrics = train_workload_model(gold_dir=gold_dir, model_dir=model_dir, random_state=random_state)
    print("Workload model training complete.")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline workload model.")
    parser.add_argument("--gold-dir", default="data/gold", help="Directory containing gold feature tables.")
    parser.add_argument("--model-dir", default="models/artifacts", help="Output directory for model artifacts.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(gold_dir=args.gold_dir, model_dir=args.model_dir, random_state=args.random_state)
