"""Rule-based schema detection — figures out target / id / tenure columns from a CSV.

The whole point of this module is so that the pipeline does not require any
hardcoded column names. After upload, app.py calls detect_schema(df) to get a
best-guess schema, surfaces it in the UI for review, and only kicks off the
pipeline once the user confirms.

Rules (deterministic — no LLM):
  * target_col: binary column whose name matches a churn-like keyword
  * id_cols:   high-cardinality columns that look like identifiers / free text
  * tenure_col: numeric column whose name suggests "time since signup"

If a rule can't find a match, the field is returned as None (or []) and the
user is expected to fill it in via the UI.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Name-based hints (case-insensitive)
# ---------------------------------------------------------------------------
_TARGET_KEYWORDS = [
    "churn", "churned", "is_churn", "has_churn", "has_churned",
    "exited", "attrition", "left", "leave", "cancelled", "canceled",
    "label", "target",
]

_TENURE_KEYWORDS = [
    "tenure", "tenure_months", "tenure_years",
    "months_active", "active_months", "account_age",
    "customer_age", "subscription_age", "days_since_signup",
    "signup_age", "membership_months",
]

_ID_NAME_PATTERNS = [
    r".*_id$", r"^id$", r".*_uuid$", r"^uuid$",
    r"^customer$", r"^user$",
]


def _norm(name: str) -> str:
    """Lowercase + strip non-alphanumerics for fuzzy name matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
def _detect_target(df: pd.DataFrame) -> str | None:
    binary_cols = [
        c for c in df.columns
        if df[c].dropna().nunique() == 2
    ]
    if not binary_cols:
        return None

    # First pass: name matches a known keyword exactly
    for col in binary_cols:
        if _norm(col) in _TARGET_KEYWORDS:
            return col

    # Second pass: name contains any keyword
    for col in binary_cols:
        norm = _norm(col)
        if any(kw in norm for kw in _TARGET_KEYWORDS):
            return col

    # Fallback: pick the binary column with the lowest minority ratio
    # (churn datasets are usually imbalanced)
    best_col = None
    best_ratio = 1.0
    for col in binary_cols:
        vals = df[col].dropna()
        counts = vals.value_counts(normalize=True)
        if len(counts) == 2:
            minority = counts.min()
            if minority < best_ratio:
                best_ratio = minority
                best_col = col
    return best_col


# ---------------------------------------------------------------------------
# Tenure column
# ---------------------------------------------------------------------------
def _detect_tenure_col(df: pd.DataFrame) -> str | None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Exact-name match first
    for col in numeric_cols:
        if _norm(col) in _TENURE_KEYWORDS:
            return col

    # Substring match
    for col in numeric_cols:
        norm = _norm(col)
        if any(kw in norm for kw in _TENURE_KEYWORDS):
            return col

    return None


# ---------------------------------------------------------------------------
# ID / non-modellable columns
# ---------------------------------------------------------------------------
def _detect_id_cols(df: pd.DataFrame) -> list[str]:
    n = len(df)
    if n == 0:
        return []

    flagged: list[str] = []
    for col in df.columns:
        norm = _norm(col)

        # Name pattern: anything ending in _id, named id/uuid, etc.
        if any(re.fullmatch(p, norm) for p in _ID_NAME_PATTERNS):
            flagged.append(col)
            continue

        # Cardinality heuristic: object/string columns where almost every
        # value is unique → free-text or identifier
        if df[col].dtype == object:
            uniq_ratio = df[col].nunique(dropna=True) / n
            if uniq_ratio > 0.9:
                flagged.append(col)

    return flagged


# ---------------------------------------------------------------------------
# Positive label
# ---------------------------------------------------------------------------
def _detect_positive_label(df: pd.DataFrame, target_col: str | None) -> Any:
    """If target is already 0/1, return 1. Otherwise return None and let
    prepare_target_node auto-detect from common patterns."""
    if not target_col or target_col not in df.columns:
        return None
    unique = set(df[target_col].dropna().unique().tolist())
    if unique <= {0, 1} or unique <= {0.0, 1.0}:
        return 1
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_schema(df: pd.DataFrame) -> dict:
    """Return a best-guess schema dict for the given dataframe.

    Always returns a dict with all four schema fields. Caller must validate
    that target_col is non-None before running the pipeline.
    """
    target_col = _detect_target(df)
    return {
        "target_col": target_col,
        "id_cols": _detect_id_cols(df),
        "tenure_col": _detect_tenure_col(df),
        "positive_label": _detect_positive_label(df, target_col),
    }
