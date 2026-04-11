import argparse
import json
from pathlib import Path

import pandas as pd


def clean_reviews(file_path: str | Path) -> pd.DataFrame:
    """
    Read raw WaniKani review statistics JSON and return a clean DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the raw review_statistics.json file

    Returns
    -------
    pd.DataFrame
        Cleaned review dataset
    """
    
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        raw_reviews = json.load(f)

    # Remove empty or malformed records that can appear in exports.
    raw_reviews = [
        record
        for record in raw_reviews
        if isinstance(record, dict) and record.get("id") is not None and isinstance(record.get("data"), dict)
    ]

    # Flatten nested "data" fields into top-level columns.
    df = pd.json_normalize(raw_reviews, sep="_")

    # Standardize timestamp columns for downstream time-based analysis.
    df["data_updated_at"] = pd.to_datetime(df["data_updated_at"], utc=True, errors="coerce")
    df["data_created_at"] = pd.to_datetime(df["data_created_at"], utc=True, errors="coerce")

    # Sort for time-based features later
    df = df.sort_values(by=["id", "data_updated_at"])

    # Keep a predictable column order: identifiers first, then stats.
    df = df[[
        "id",
        "object",
        "url",
        "data_updated_at",
        "data_created_at",
        "data_subject_id",
        "data_subject_type",
        "data_meaning_correct",
        "data_meaning_incorrect",
        "data_meaning_max_streak",
        "data_meaning_current_streak",
        "data_reading_correct",
        "data_reading_incorrect",
        "data_reading_max_streak",
        "data_reading_current_streak",
        "data_percentage_correct",
        "data_hidden",
    ]]
    
    return df


flattened_review_data = clean_reviews("data/raw/review_statistics.json")
print(flattened_review_data.head())