import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_TARGET_COL = "Depression (1 - Yes, 0 - No)"


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {
        "Gender (0 - Male, 1 - Female)": "Gender",
        "Age": "Age",
        "Academic Pressure": "Academic_Pressure",
        "Study Satisfaction": "Study_Satisfaction",
        "Sleep Duration (Hours)": "Sleep_Duration",
        "Dietary Habits (0 - Healthy, 1 - Moderate, 2 - Unhealthy)": "Dietary_Habits",
        "Have you ever had suicidal thoughts ? (1 - 1, 0 - 0)": "Suicidal_Thoughts",
        "Study Hours": "Study_Hours",
        "Financial Stress": "Financial_Stress",
        "Family History of Mental Illness (1 -Yes, 0 - No)": "Family_History",
        RAW_TARGET_COL: "Depression",
    }
    cols = [c.strip() for c in df.columns]
    df.columns = [mapping.get(c, c) for c in cols]
    return df


def _coerce_sleep_duration(series: pd.Series) -> pd.Series:
    # Clean spaces and unify tokens like "< 5" -> "<5", "7-8 " -> "7-8"
    tokens = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    mapping = {
        "<5": 4.5,
        "5-6": 5.5,
        "7-8": 7.5,
        ">8": 9.0,
    }
    numeric = tokens.map(mapping)
    # If some entries are actual numerics, try to parse
    numeric = numeric.where(~numeric.isna(), pd.to_numeric(tokens, errors="coerce"))
    return numeric


def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df = _normalize_headers(df)

    # Coerce dtypes
    if "Sleep_Duration" in df.columns:
        df["Sleep_Duration"] = _coerce_sleep_duration(df["Sleep_Duration"]) 

    # Ensure numeric types for all features where applicable
    numeric_cols: List[str] = [
        "Gender",
        "Age",
        "Academic_Pressure",
        "Study_Satisfaction",
        "Sleep_Duration",
        "Dietary_Habits",
        "Suicidal_Thoughts",
        "Study_Hours",
        "Financial_Stress",
        "Family_History",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target
    if "Depression" not in df.columns:
        raise ValueError("Target column 'Depression' not found after header normalization.")
    df = df.dropna(subset=["Depression"])  # ensure target present

    # Simple imputation for numeric features
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[numeric_cols]
    y = df["Depression"].astype(int)
    return X, y


def train_test_split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_feature_info(path: str | Path, feature_names: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_names}, f, indent=2)


