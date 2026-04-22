import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path):
    """Load and parse a JSON file from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_resource_collection(records: list[dict]) -> pd.DataFrame:
    """Filter valid API resources and flatten them into a DataFrame."""
    records = [
        r for r in records
        if isinstance(r, dict) and isinstance(r.get("data"), dict) and r.get("id") is not None
    ]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep="_")


def build_assignments_silver(raw_dir: Path) -> pd.DataFrame:
    """Create a normalized assignments table from the raw assignments export."""
    raw = _load_json(raw_dir / "assignments.json")
    df = _normalize_resource_collection(raw)
    if df.empty:
        return df

    for col in [
        "data_updated_at", "data_created_at", "data_unlocked_at", "data_started_at",
        "data_passed_at", "data_burned_at", "data_available_at", "data_resurrected_at",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    selected = [
        "id", "object", "url", "data_updated_at", "data_created_at", "data_subject_id",
        "data_subject_type", "data_level", "data_srs_stage", "data_unlocked_at", "data_started_at",
        "data_passed_at", "data_burned_at", "data_available_at", "data_resurrected_at", "data_hidden",
    ]
    selected = [c for c in selected if c in df.columns]
    return df[selected].rename(columns={"id": "assignment_id", "data_subject_id": "subject_id"})


def build_review_stats_silver(raw_dir: Path) -> pd.DataFrame:
    """Create a normalized review statistics table from the raw export."""
    raw = _load_json(raw_dir / "review_statistics.json")
    df = _normalize_resource_collection(raw)
    if df.empty:
        return df

    for col in ["data_updated_at", "data_created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    selected = [
        "id", "object", "url", "data_updated_at", "data_created_at", "data_subject_id",
        "data_subject_type", "data_meaning_correct", "data_meaning_incorrect",
        "data_meaning_max_streak", "data_meaning_current_streak", "data_reading_correct",
        "data_reading_incorrect", "data_reading_max_streak", "data_reading_current_streak",
        "data_percentage_correct", "data_hidden",
    ]
    selected = [c for c in selected if c in df.columns]
    return df[selected].rename(columns={"id": "review_stat_id", "data_subject_id": "subject_id"})


def build_subjects_silver(raw_dir: Path) -> pd.DataFrame:
    """Create a normalized subjects table and derive simple content features."""
    raw = _load_json(raw_dir / "subjects.json")
    df = _normalize_resource_collection(raw)
    if df.empty:
        return df

    for col in ["data_updated_at", "data_created_at", "data_hidden_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    def _safe_len(x):
        return len(x) if isinstance(x, list) else 0

    def _primary_meaning(meanings):
        if not isinstance(meanings, list):
            return None
        for m in meanings:
            if isinstance(m, dict) and m.get("primary"):
                return m.get("meaning")
        return None

    df["primary_meaning"] = df.get("data_meanings", pd.Series(index=df.index)).apply(_primary_meaning)
    df["has_reading"] = df.get("data_readings", pd.Series(index=df.index)).apply(lambda x: int(_safe_len(x) > 0))
    df["has_context_sentences"] = df.get("data_context_sentences", pd.Series(index=df.index)).apply(lambda x: int(_safe_len(x) > 0))
    df["component_subject_count"] = df.get("data_component_subject_ids", pd.Series(index=df.index)).apply(_safe_len)
    df["amalgamation_subject_count"] = df.get("data_amalgamation_subject_ids", pd.Series(index=df.index)).apply(_safe_len)
    df["visually_similar_count"] = df.get("data_visually_similar_subject_ids", pd.Series(index=df.index)).apply(_safe_len)
    df["parts_of_speech_count"] = df.get("data_parts_of_speech", pd.Series(index=df.index)).apply(_safe_len)

    selected = [
        "id", "object", "url", "data_updated_at", "data_created_at", "data_level", "data_slug",
        "data_characters", "primary_meaning", "data_lesson_position", "data_spaced_repetition_system_id", "data_hidden_at",
        "has_reading", "has_context_sentences", "component_subject_count", "amalgamation_subject_count",
        "visually_similar_count", "parts_of_speech_count",
    ]
    selected = [c for c in selected if c in df.columns]
    return df[selected].rename(columns={"id": "subject_id", "object": "subject_type"})


def build_summary_hourly_silver(raw_dir: Path) -> pd.DataFrame:
    """Aggregate summary report lessons and reviews into hourly counts."""
    raw = _load_json(raw_dir / "summary.json")
    # The pull script saves the unwrapped data object directly, so raw IS the report.
    # Fall back to raw["data"] for compatibility with any legacy format.
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        report = raw["data"]
    elif isinstance(raw, dict) and ("reviews" in raw or "lessons" in raw):
        report = raw
    else:
        return pd.DataFrame()

    rows = []

    for row in report.get("reviews", []):
        rows.append(
            {
                "available_at_hour": row.get("available_at"),
                "review_count": len(row.get("subject_ids", [])),
                "lesson_count": 0,
                "next_reviews_at": report.get("next_reviews_at"),
            }
        )

    for row in report.get("lessons", []):
        rows.append(
            {
                "available_at_hour": row.get("available_at"),
                "review_count": 0,
                "lesson_count": len(row.get("subject_ids", [])),
                "next_reviews_at": report.get("next_reviews_at"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["available_at_hour"] = pd.to_datetime(df["available_at_hour"], utc=True, errors="coerce")
    df["next_reviews_at"] = pd.to_datetime(df["next_reviews_at"], utc=True, errors="coerce")
    return df.groupby("available_at_hour", as_index=False).agg(
        review_count=("review_count", "sum"),
        lesson_count=("lesson_count", "sum"),
        next_reviews_at=("next_reviews_at", "max"),
    )


def main(raw_dir: str, silver_dir: str) -> None:
    """Build all silver tables and write them to the configured output directory."""
    raw_path = Path(raw_dir)
    silver_path = Path(silver_dir)
    silver_path.mkdir(parents=True, exist_ok=True)

    assignments = build_assignments_silver(raw_path)
    review_stats = build_review_stats_silver(raw_path)
    subjects = build_subjects_silver(raw_path)
    summary = build_summary_hourly_silver(raw_path)

    assignments.to_csv(silver_path / "assignments_silver.csv", index=False)
    review_stats.to_csv(silver_path / "review_stats_silver.csv", index=False)
    subjects.to_csv(silver_path / "subjects_silver.csv", index=False)
    summary.to_csv(silver_path / "summary_hourly_silver.csv", index=False)

    print("Saved silver tables:")
    print(f"- {silver_path / 'assignments_silver.csv'} ({len(assignments)} rows)")
    print(f"- {silver_path / 'review_stats_silver.csv'} ({len(review_stats)} rows)")
    print(f"- {silver_path / 'subjects_silver.csv'} ({len(subjects)} rows)")
    print(f"- {silver_path / 'summary_hourly_silver.csv'} ({len(summary)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build silver tables from raw WaniKani exports.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw endpoint JSON files.")
    parser.add_argument("--silver-dir", default="data/silver", help="Output directory for silver CSV files.")
    args = parser.parse_args()

    main(raw_dir=args.raw_dir, silver_dir=args.silver_dir)
