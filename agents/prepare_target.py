"""Target encoding node — runs first, encodes the schema's target column to 0/1.

Most downstream agents assume the target is binary 0/1 (class_imbalance,
horizon, model training). This node normalises it regardless of how the
raw dataset stores the label (e.g. "Yes"/"No", True/False, 1/0).

Encoding rules:
  1. If schema["positive_label"] is provided, target == positive_label  → 1, else → 0
  2. Otherwise try common patterns: Yes/No, True/False, Y/N, 1/0
  3. If the column already contains only 0s and 1s, leave it alone

Failure to encode is a fatal error — surfaced via state["progress_messages"].
"""

from agents.state import PipelineState


_KNOWN_TRUTHY = {"yes", "y", "true", "t", "1", "churned", "churn"}
_KNOWN_FALSY = {"no", "n", "false", "f", "0", "active", "retained", "stayed"}


def prepare_target_node(state: PipelineState) -> dict:
    df = state["raw_df"].copy()
    schema = state["schema"]
    target_col = schema["target_col"]
    positive_label = schema.get("positive_label")
    msgs = list(state.get("progress_messages", []))

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    series = df[target_col]
    unique = set(series.dropna().unique().tolist())

    # Already 0/1 → done
    if unique <= {0, 1} or unique <= {0.0, 1.0}:
        df[target_col] = series.astype(int)
        msgs.append(f"Target '{target_col}' already 0/1 — no encoding needed.")
        return {"raw_df": df, "current_step": "target_prepared", "progress_messages": msgs}

    # Explicit positive_label
    if positive_label is not None and positive_label in unique:
        df[target_col] = (series == positive_label).astype(int)
        msgs.append(f"Encoded '{target_col}': {positive_label!r} → 1, others → 0")
        return {"raw_df": df, "current_step": "target_prepared", "progress_messages": msgs}

    # Try common truthy/falsy patterns
    str_unique = {str(v).strip().lower() for v in unique}
    truthy_match = str_unique & _KNOWN_TRUTHY
    falsy_match = str_unique & _KNOWN_FALSY

    if truthy_match and falsy_match:
        df[target_col] = series.map(
            lambda v: 1 if str(v).strip().lower() in _KNOWN_TRUTHY else 0
        ).astype(int)
        msgs.append(
            f"Encoded '{target_col}': "
            f"{sorted(truthy_match)} → 1, {sorted(falsy_match)} → 0"
        )
        return {"raw_df": df, "current_step": "target_prepared", "progress_messages": msgs}

    raise ValueError(
        f"Cannot auto-encode target column '{target_col}' (unique values: {sorted(map(str, unique))}). "
        f"Set schema['positive_label'] explicitly."
    )
