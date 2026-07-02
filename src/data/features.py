"""Feature preparation shared by the uplift notebooks and tests.

Kept deliberately simple: the covariates are pre-treatment customer
attributes, so there is no leakage risk from the outcome. The only real
subtlety is that LightGBM rejects special characters in feature names
(the ``history_segment`` labels contain ``$`` and ``()``), so column names
are sanitised after one-hot encoding.
"""

from __future__ import annotations

import re

import pandas as pd

FEATURE_COLS = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "history_segment",
    "zip_code",
    "channel",
]


def _sanitise(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and return a numeric feature matrix with
    LightGBM-safe column names."""
    X = pd.get_dummies(df[FEATURE_COLS], drop_first=True)
    X.columns = [_sanitise(c) for c in X.columns]
    return X
