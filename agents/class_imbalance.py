"""Node 4 — Class Imbalance Agent: detect target imbalance and configure mitigation."""

from agents.state import PipelineState


def class_imbalance_node(state: PipelineState) -> dict:
    df = state["raw_df"]
    target_col = state["schema"]["target_col"]
    target = df[target_col]

    n_total = len(target)
    n_minority = int(target.sum())
    n_majority = n_total - n_minority
    minority_ratio = n_minority / n_total if n_total > 0 else 0.5

    is_imbalanced = minority_ratio < 0.20

    imbalance_config = {
        "minority_ratio": round(float(minority_ratio), 4),
        "minority_count": n_minority,
        "majority_count": n_majority,
        "is_imbalanced": is_imbalanced,
        # Use PR-AUC as primary CV metric when data is imbalanced (ROC-AUC is over-optimistic).
        # Keep class weights neutral so predict_proba remains usable as business probability.
        "primary_metric": "average_precision" if is_imbalanced else "roc_auc",
        "logreg_class_weight": None,
        "rf_class_weight": None,
        "lgbm_class_weight": None,
        "xgb_scale_pos_weight": 1.0,
    }

    status = "imbalanced" if is_imbalanced else "balanced"
    msg = (
        f"Class ratio: {minority_ratio:.1%} minority "
        f"({n_minority:,} positive / {n_total:,} total) — {status}"
    )
    if is_imbalanced:
        msg += (
            ". Mitigation: CV metric=average_precision; class weights left neutral "
            "to keep churn probabilities calibrated for revenue estimates"
        )

    return {
        "imbalance_config": imbalance_config,
        "current_step": "class_imbalance",
        "progress_messages": state.get("progress_messages", []) + [msg],
    }
