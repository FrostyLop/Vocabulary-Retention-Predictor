import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_risk_model(gold_dir: str | Path, model_dir: str | Path, random_state: int = 42) -> dict:
    gold_dir = Path(gold_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(gold_dir / "subject_features_daily.csv")
    labels = pd.read_csv(gold_dir / "risk_labels_daily.csv")

    data = features.merge(labels[["subject_id", "risk_label_binary"]], on="subject_id", how="inner")
    if data.empty:
        raise ValueError("No training rows found. Build gold features first.")

    y = data["risk_label_binary"].astype(int)
    drop_cols = ["risk_label_binary", "snapshot_date", "subject_id"]
    X = data.drop(columns=[c for c in drop_cols if c in data.columns])

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    artifact_path = model_dir / "risk_model.joblib"
    joblib.dump(model, artifact_path)

    metrics = {
        "roc_auc": float(auc),
        "precision_high_risk": float(report.get("1", {}).get("precision", 0.0)),
        "recall_high_risk": float(report.get("1", {}).get("recall", 0.0)),
        "f1_high_risk": float(report.get("1", {}).get("f1-score", 0.0)),
        "support_high_risk": int(report.get("1", {}).get("support", 0)),
        "artifact_path": str(artifact_path),
    }

    pd.DataFrame([metrics]).to_csv(model_dir / "risk_model_metrics.csv", index=False)
    return metrics


def main(gold_dir: str, model_dir: str, random_state: int) -> None:
    metrics = train_risk_model(gold_dir=gold_dir, model_dir=model_dir, random_state=random_state)
    print("Risk model training complete.")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline subject risk model.")
    parser.add_argument("--gold-dir", default="data/gold", help="Directory containing gold feature tables.")
    parser.add_argument("--model-dir", default="models/artifacts", help="Output directory for model artifacts.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(gold_dir=args.gold_dir, model_dir=args.model_dir, random_state=args.random_state)
