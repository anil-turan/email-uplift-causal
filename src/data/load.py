"""Load and validate the Hillstrom e-mail marketing dataset.

The dataset is a genuine randomised experiment run by Kevin Hillstrom
(MineThatData, 2008). 64,000 customers who had purchased in the last
twelve months were randomised into three arms:

    - "Mens E-Mail"   : sent an e-mail campaign featuring men's merchandise
    - "Womens E-Mail" : sent an e-mail campaign featuring women's merchandise
    - "No E-Mail"     : control, received nothing

Outcomes recorded over the following two weeks:
    - visit      : did the customer visit the site? (0/1)
    - conversion : did the customer purchase? (0/1)
    - spend      : amount spent (float, mostly zero)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hillstrom.csv"

TREATMENT_COL = "segment"
COVARIATES_NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
COVARIATES_CATEGORICAL = ["history_segment", "zip_code", "channel"]
OUTCOMES = ["visit", "conversion", "spend"]

ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and run integrity checks that would fail loudly
    if the file were swapped or corrupted."""
    df = pd.read_csv(path)

    expected = set(
        [TREATMENT_COL, *COVARIATES_NUMERIC, *COVARIATES_CATEGORICAL, *OUTCOMES]
    )
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    unexpected_arms = set(df[TREATMENT_COL].unique()) - set(ARMS)
    if unexpected_arms:
        raise ValueError(f"Unexpected treatment arms: {unexpected_arms}")

    if df[OUTCOMES].isna().any().any():
        raise ValueError("Outcome columns contain missing values")

    return df


def add_treatment_flag(df: pd.DataFrame, drop_arm: str = "Womens E-Mail") -> pd.DataFrame:
    """Collapse the three arms into a binary treatment for the headline
    experiment analysis: Mens E-Mail (treatment=1) vs No E-Mail (control=0).

    The Womens arm is dropped by default so the primary comparison is a clean
    two-arm A/B test. The full three-arm data is kept for uplift modeling.
    """
    two_arm = df[df[TREATMENT_COL] != drop_arm].copy()
    two_arm["treatment"] = (two_arm[TREATMENT_COL] == "Mens E-Mail").astype(int)
    return two_arm


def full_treatment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary treatment where ANY e-mail is treatment=1, No E-Mail is control=0.
    Used for uplift modeling where we want the effect of 'send an e-mail'."""
    out = df.copy()
    out["treatment"] = (out[TREATMENT_COL] != "No E-Mail").astype(int)
    return out
