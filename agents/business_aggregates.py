"""Business aggregates node — BA-friendly numbers computed from model outputs.

This is a pure-Python node (no LLM). It turns the model's raw test-set predictions
into the sort of numbers a retention manager actually asks about:

  * How many customers are predicted to churn?  (at optimal threshold)
  * What's the revenue at stake?                (at-risk count × customer_value)
  * If we run a retention campaign, what's the expected profit?
  * Break the customer base into high/medium/low-risk buckets
  * Who are the top-N highest-risk individuals? (for the Who's at Risk tab)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.state import PipelineState
from pipeline.config import BUSINESS_CONSTANTS
from pipeline.tasks import PrepareData


# Risk bucket thresholds on predicted probability.
_BUCKET_EDGES = {"low": (0.0, 0.30), "medium": (0.30, 0.60), "high": (0.60, 1.01)}

# How many top-risk individual customers to surface in the UI.
_TOP_N_AT_RISK = 50


def _classify_buckets(probs: np.ndarray) -> dict:
    buckets = {}
    for name, (lo, hi) in _BUCKET_EDGES.items():
        buckets[name] = int(((probs >= lo) & (probs < hi)).sum())
    return buckets


def business_aggregates_node(state: PipelineState) -> dict:
    best = state.get("best_model_metrics") or {}
    preds = state.get("predictions") or {}

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
    threshold = float(best.get("optimal_threshold", 0.5))

    # ── Predicted at-risk count (at the model's optimal threshold) ──
    at_risk_mask = y_prob >= threshold
    at_risk_count = int(at_risk_mask.sum())
    at_risk_pct = round(float(at_risk_mask.mean()) * 100, 1)

    # ── Revenue at stake & projected profit ──
    cv = BUSINESS_CONSTANTS["customer_value"]
    revenue_at_stake = float(at_risk_count * cv)
    projected_profit = float(best.get("expected_profit", 0.0))

    # ── Risk buckets (low / medium / high) ──
    risk_bucket_counts = _classify_buckets(y_prob)

    # ── Top-N individual at-risk customers ──
    # We need the feature rows for the highest-probability customers. Load the
    # test set from the d6tflow cache that model_selection produced.
    try:
        data = PrepareData().output().load()
        X_test: pd.DataFrame = data["X_test"].reset_index(drop=True).copy()
        X_test["_churn_probability"] = y_prob
        X_test["_actual_churn"] = y_test
        top_rows = (
            X_test.sort_values("_churn_probability", ascending=False)
                  .head(_TOP_N_AT_RISK)
                  .to_dict(orient="records")
        )
    except Exception:
        top_rows = []

    aggregates = {
        "at_risk_count": at_risk_count,
        "at_risk_pct": at_risk_pct,
        "revenue_at_stake": round(revenue_at_stake, 2),
        "projected_profit": round(projected_profit, 2),
        "threshold_used": round(threshold, 3),
        "customer_value": cv,
        "risk_bucket_counts": risk_bucket_counts,
        "top_at_risk_customers": top_rows,
        "test_set_size": int(len(y_prob)),
    }

    msg = (
        f"Aggregates: {at_risk_count} customers at risk ({at_risk_pct}% of test set), "
        f"${revenue_at_stake:,.0f} revenue at stake, projected profit ${projected_profit:,.0f}"
    )

    return {
        "business_aggregates": aggregates,
        "current_step": "business_aggregates",
        "progress_messages": state.get("progress_messages", []) + [msg],
    }
