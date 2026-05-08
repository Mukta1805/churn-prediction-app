"""Business aggregates node — BA-friendly numbers computed from model outputs.

This is a pure-Python node (no LLM). It turns the model's raw test-set predictions
into the sort of numbers a retention manager actually asks about.

Three distinct concepts are kept separate (they were previously conflated):

  * AT-RISK = customers with elevated churn probability (default ≥ 0.50).
    Answers: "who has high churn risk?"  Threshold-of-the-data, not of the action.

  * CONTACT LIST = customers above the model's profit-optimal threshold.
    Answers: "who is profitable to contact under the current business assumptions?"

  * REVENUE AT STAKE = sum of the dataset's value column for at-risk customers
    when a value column is detected on the schema; otherwise falls back to
    at_risk_count × user-supplied customer_value (BA owns that assumption).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.state import PipelineState
from pipeline.config import BUSINESS_CONSTANTS
from pipeline.tasks import PrepareData


# Risk bucket thresholds on predicted probability.
_BUCKET_EDGES = {"low": (0.0, 0.30), "medium": (0.30, 0.60), "high": (0.60, 1.01)}

# Default "at-risk" cutoff. 0.50 = "more likely than not to churn" — universally
# meaningful and BA-intuitive across base rates and model calibrations.
_AT_RISK_THRESHOLD = 0.50

# How many top-risk individual customers to surface in the UI.
_TOP_N_AT_RISK = 50


def _classify_buckets(probs: np.ndarray) -> dict:
    buckets = {}
    for name, (lo, hi) in _BUCKET_EDGES.items():
        buckets[name] = int(((probs >= lo) & (probs < hi)).sum())
    return buckets


def _compute_revenue_at_stake(
    at_risk_mask: np.ndarray,
    X_test: pd.DataFrame | None,
    value_col: str | None,
    customer_value: float,
) -> tuple[float, str]:
    """Returns (revenue_at_stake, methodology_note).

    Strategy: if a value column is on the schema and present in X_test,
    sum its values for at-risk rows (truthful). Otherwise fall back to the
    placeholder count × CLV (BA owns that number via the sidebar).
    """
    if (
        value_col
        and X_test is not None
        and value_col in X_test.columns
    ):
        values = pd.to_numeric(X_test[value_col], errors="coerce").fillna(0.0).to_numpy()
        revenue = float(values[at_risk_mask].sum())
        return revenue, f"sum of `{value_col}` for at-risk customers"

    revenue = float(int(at_risk_mask.sum()) * customer_value)
    return revenue, f"at-risk count × ${customer_value:,.0f} (placeholder — no value column detected)"


def business_aggregates_node(state: PipelineState) -> dict:
    best = state.get("best_model_metrics") or {}
    preds = state.get("predictions") or {}
    schema = state.get("schema") or {}

    if not preds or "y_prob" not in preds:
        return {
            "business_aggregates": {},
            "current_step": "business_aggregates (skipped — no predictions)",
            "progress_messages": state.get("progress_messages", []) + [
                "Skipped business aggregates: no test-set predictions available"
            ],
        }

    y_prob = np.array(preds["y_prob"])
    y_test = np.array(preds["y_test"])
    optimal_threshold = float(best.get("optimal_threshold", 0.5))

    # ── AT-RISK (risk classification, not action) ──
    at_risk_mask = y_prob >= _AT_RISK_THRESHOLD
    at_risk_count = int(at_risk_mask.sum())
    at_risk_pct = round(float(at_risk_mask.mean()) * 100, 1)

    # ── CONTACT LIST (action recommendation per profit math) ──
    contact_mask = y_prob >= optimal_threshold
    contact_list_count = int(contact_mask.sum())
    contact_list_pct = round(float(contact_mask.mean()) * 100, 1)

    # ── Load the prepared X_test once — used for revenue at stake AND top-N ──
    try:
        data = PrepareData().output().load()
        X_test: pd.DataFrame = data["X_test"].reset_index(drop=True).copy()
    except Exception:
        X_test = None

    # ── REVENUE AT STAKE (real value column when available) ──
    customer_value = float(state.get("customer_value", BUSINESS_CONSTANTS["customer_value"]))
    value_col = schema.get("value_col")
    revenue_at_stake, revenue_methodology = _compute_revenue_at_stake(
        at_risk_mask=at_risk_mask,
        X_test=X_test,
        value_col=value_col,
        customer_value=customer_value,
    )

    # ── Projected profit (unchanged — driven by profit-optimal threshold) ──
    projected_profit = float(best.get("expected_profit", 0.0))

    # ── Risk buckets (low / medium / high) ──
    risk_bucket_counts = _classify_buckets(y_prob)

    # ── Top-N individual at-risk customers ──
    top_rows: list[dict] = []
    if X_test is not None:
        try:
            X_show = X_test.copy()
            X_show["_churn_probability"] = y_prob
            X_show["_actual_churn"] = y_test
            top_rows = (
                X_show.sort_values("_churn_probability", ascending=False)
                      .head(_TOP_N_AT_RISK)
                      .to_dict(orient="records")
            )
        except Exception:
            top_rows = []

    aggregates = {
        # At-risk (risk classification)
        "at_risk_count": at_risk_count,
        "at_risk_pct": at_risk_pct,
        "at_risk_threshold": _AT_RISK_THRESHOLD,
        # Contact list (action under profit math)
        "contact_list_count": contact_list_count,
        "contact_list_pct": contact_list_pct,
        "optimal_threshold": round(optimal_threshold, 3),
        # Revenue
        "revenue_at_stake": round(revenue_at_stake, 2),
        "revenue_methodology": revenue_methodology,
        "value_col": value_col,
        "customer_value": customer_value,
        # Profit
        "projected_profit": round(projected_profit, 2),
        # Buckets
        "risk_bucket_counts": risk_bucket_counts,
        "top_at_risk_customers": top_rows,
        "test_set_size": int(len(y_prob)),
        # Kept for backwards compatibility — exec-summary captions still read this
        "threshold_used": round(optimal_threshold, 3),
    }

    msg = (
        f"Aggregates: {at_risk_count} at-risk customers (≥{_AT_RISK_THRESHOLD:.0%} prob), "
        f"{contact_list_count} on contact list (≥{optimal_threshold:.2f} prob), "
        f"${revenue_at_stake:,.0f} revenue at stake ({revenue_methodology}), "
        f"projected profit ${projected_profit:,.0f}"
    )

    return {
        "business_aggregates": aggregates,
        "current_step": "business_aggregates",
        "progress_messages": state.get("progress_messages", []) + [msg],
    }
