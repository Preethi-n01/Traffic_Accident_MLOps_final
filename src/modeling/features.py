from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "collision_severity"

LEAKAGE_COLUMNS = [
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
]

IDENTIFIER_COLUMNS = [
    "collision_index",
    "collision_ref_no",
    "lsoa_of_accident_location",
]


def load_labeled_data(data_path: Path) -> pd.DataFrame:
    """Load labeled collisions in chronological order."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return (
        df.dropna(subset=[TARGET, "date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same outcome-safe feature selection before model inference."""
    features = df.drop(
        columns=[TARGET, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS, "date", "time"],
        errors="ignore",
    ).copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    for column in features.select_dtypes(include=["object", "string"]).columns:
        features[column] = features[column].astype("string").fillna("__MISSING__")
    return features
