"""Horizon definition tools — synthetic time construction and label generation.

Both functions take the relevant column names as arguments so they work on any
dataset, not just the original customer churn one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_time(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    tenure_col: str,
    target_col: str = "churn",
) -> pd.DataFrame:
    """Add start_date, days_before_snapshot, and churn_date columns.

    Assumes df[target_col] is already encoded as 0/1 and df[tenure_col] is
    numeric (interpreted as months).
    """
    df = df.copy()

    df["start_date"] = snapshot_date - pd.to_timedelta(df[tenure_col] * 30, unit="D")

    np.random.seed(42)
    df["days_before_snapshot"] = np.where(
        df[target_col] == 1,
        np.random.randint(0, 90, size=len(df)),
        np.nan,
    )

    df["churn_date"] = np.where(
        df[target_col] == 1,
        snapshot_date - pd.to_timedelta(df["days_before_snapshot"], unit="D"),
        pd.NaT,
    )

    return df


def build_horizon_labels(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    horizons: list[int] | None = None,
    target_col: str = "churn",
) -> pd.DataFrame:
    """Add {target_col}_{h}d binary columns for each horizon h."""
    if horizons is None:
        horizons = [30, 60, 90]

    df = df.copy()
    df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce")
    df["days_since_churn"] = (snapshot_date - df["churn_date"]).dt.days

    for h in horizons:
        df[f"{target_col}_{h}d"] = (
            (df[target_col] == 1) & (df["days_since_churn"] <= h)
        ).astype(int)

    return df
