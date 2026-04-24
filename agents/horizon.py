"""Horizon definition node — Node 2 in the offline pipeline.

Responsibilities:
- Generate synthetic time structure (start_date, churn_date) using the schema's
  tenure column
- Build horizon labels: {target}_30d, {target}_60d, {target}_90d
- Set the active target column to the selected horizon's binary
- Drop leakage columns before passing data downstream
- Preserve df_master (with all horizon labels intact) for later reference

Skipped automatically by graph routing if state["schema"]["tenure_col"] is None.
"""

from agents.state import PipelineState
from pipeline.config import DEFAULT_HORIZON, HORIZONS, SNAPSHOT_DATE
from utils.horizon_utils import build_horizon_labels, generate_synthetic_time


def horizon_definition_node(state: PipelineState) -> dict:
    df = state["raw_df"].copy()
    schema = state["schema"]
    target_col = schema["target_col"]
    tenure_col = schema["tenure_col"]

    horizon = state.get("selected_horizon") or DEFAULT_HORIZON
    messages = list(state.get("progress_messages", []))

    df = generate_synthetic_time(df, SNAPSHOT_DATE, tenure_col=tenure_col, target_col=target_col)
    messages.append(f"Generated synthetic time structure (snapshot: {SNAPSHOT_DATE.date()})")

    df = build_horizon_labels(df, SNAPSHOT_DATE, HORIZONS, target_col=target_col)
    label_cols = ", ".join(f"{target_col}_{h}d" for h in HORIZONS)
    messages.append(f"Built horizon labels: {label_cols}")

    # Keep df_master before any leakage columns are dropped
    df_master = df.copy()

    # Overwrite target with the selected horizon's label
    df[target_col] = df[f"{target_col}_{horizon}d"]
    messages.append(f"Active target: {target_col}_{horizon}d ({horizon}-day window)")

    # Drop columns that would leak future information into the model
    leakage_cols = [
        "start_date",
        "churn_date",
        "days_before_snapshot",
        "days_since_churn",
    ] + [f"{target_col}_{h}d" for h in HORIZONS]
    drop_cols = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    return {
        "raw_df": df,
        "df_master": df_master,
        "selected_horizon": horizon,
        "current_step": f"Horizon defined ({horizon}d)",
        "progress_messages": messages,
    }
