"""Segment discovery — decision-tree surrogate + LLM naming/narration.

The idea: train a small decision tree (max_depth=3) on the best model's predicted
probabilities. Each leaf becomes a customer segment defined by a rule. Then feed
the rules + leaf statistics to the LLM so it produces a business-friendly name,
narrative, and recommended actions per segment.

Tree surrogate is deterministic and auditable — the rules come directly from
the data. The LLM only converts facts into language, it doesn't invent anything.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.tree import DecisionTreeRegressor, _tree

from agents.state import PipelineState
from pipeline.tasks import PrepareData


_MAX_TREE_DEPTH = 3                  # controls how many segments we produce (≤ 8 leaves)
_MIN_LEAF_SIZE = 30                  # small segments collapse into the parent
_MIN_CHURN_PROB_FOR_SEGMENT = 0.15   # drop leaves with near-zero churn prob (they're not interesting)


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Tree → list of leaf rules
# ---------------------------------------------------------------------------
def _tree_to_rules(tree, feature_names: list[str]) -> list[dict]:
    """Walk a fitted DecisionTreeRegressor and return one dict per leaf.

    Each dict contains:
      - rule:   list of (feature, operator, threshold) tuples
      - value:  predicted churn probability at this leaf
      - samples: number of training rows falling into this leaf
    """
    t = tree.tree_
    leaves = []

    def recurse(node_id: int, conditions: list[tuple]):
        if t.feature[node_id] == _tree.TREE_UNDEFINED:
            # Leaf
            leaves.append({
                "rule": list(conditions),
                "value": float(t.value[node_id][0][0]),
                "samples": int(t.n_node_samples[node_id]),
            })
            return

        feature = feature_names[t.feature[node_id]]
        threshold = float(t.threshold[node_id])
        recurse(t.children_left[node_id], conditions + [(feature, "<=", threshold)])
        recurse(t.children_right[node_id], conditions + [(feature, ">", threshold)])

    recurse(0, [])
    return leaves


def _rule_to_human(rule: list[tuple]) -> str:
    """Convert a list of (feature, op, threshold) tuples into a readable string."""
    parts = []
    for feat, op, val in rule:
        # Pretty formatting: one-hot columns "Geography_Germany <= 0.5" → "Geography != Germany"
        if "_" in feat and val == 0.5:
            base, _, cat = feat.partition("_")
            if op == "<=":
                parts.append(f"{base} ≠ {cat}")
            else:
                parts.append(f"{base} = {cat}")
        else:
            num_val = f"{val:.2f}" if abs(val) < 1000 else f"{val:,.0f}"
            parts.append(f"{feat} {op} {num_val}")
    return " AND ".join(parts)


# ---------------------------------------------------------------------------
# LLM naming / narration
# ---------------------------------------------------------------------------
_SEGMENT_PROMPT = """\
You are a retention analyst summarising customer segments for a business audience.

Below are customer segments discovered from a churn-prediction model. For each segment,
return:
  - "name": a short, memorable name (2-5 words, title case)
  - "characteristics": 2-3 bullet-style strings describing the segment in plain English
  - "narrative": 1-2 sentence business explanation of who these customers are and why they're at risk
  - "recommended_actions": 2-3 concrete retention plays tailored to this segment

Segments:
{segments_json}

Return a JSON object with one top-level field "segments", which is a list of objects in the
same order as the input. Each object must contain the four fields above. Do NOT change the
rules, sizes, or churn rates.
"""


def _call_llm_for_names(segment_facts: list[dict]) -> list[dict]:
    payload = [
        {
            "rule": _rule_to_human(s["rule"]),
            "size": s["samples"],
            "predicted_churn_rate": round(s["value"], 3),
        }
        for s in segment_facts
    ]
    prompt = _SEGMENT_PROMPT.format(segments_json=json.dumps(payload, indent=2))

    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("segments", [])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def segment_discovery_node(state: PipelineState) -> dict:
    best_pipeline = state.get("best_pipeline")
    preds = state.get("predictions") or {}

    if best_pipeline is None or "y_prob" not in preds:
        return {
            "segments": [],
            "current_step": "segment_discovery (skipped)",
            "progress_messages": state.get("progress_messages", []) + [
                "Skipped segment discovery: no fitted pipeline or predictions"
            ],
        }

    # ── Load test data & transform through the same preprocessor ──
    data = PrepareData().output().load()
    X_test = data["X_test"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_test)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    feature_names = state.get("feature_names") or []
    if not feature_names:
        feature_names = [f"f{i}" for i in range(X_proc.shape[1])]

    y_prob = np.array(preds["y_prob"])

    # ── Fit the surrogate tree on predicted probability ──
    tree = DecisionTreeRegressor(
        max_depth=_MAX_TREE_DEPTH,
        min_samples_leaf=_MIN_LEAF_SIZE,
        random_state=42,
    )
    tree.fit(X_proc, y_prob)

    leaves = _tree_to_rules(tree, feature_names)

    # Drop boring leaves (too small, or with low churn rate)
    n_total = len(y_prob)
    interesting = [
        leaf for leaf in leaves
        if leaf["samples"] >= _MIN_LEAF_SIZE and leaf["value"] >= _MIN_CHURN_PROB_FOR_SEGMENT
    ]
    # Keep at most the top 6 by predicted churn rate
    interesting.sort(key=lambda s: s["value"], reverse=True)
    interesting = interesting[:6]

    if not interesting:
        # Fallback: take the top-2 leaves even if low-churn, so the BA sees something
        leaves.sort(key=lambda s: s["value"], reverse=True)
        interesting = leaves[:2]

    # ── Ask the LLM to name + narrate each segment ──
    try:
        llm_segments = _call_llm_for_names(interesting)
    except Exception as e:
        llm_segments = [{} for _ in interesting]
        narration_error = str(e)
    else:
        narration_error = None

    # ── Compose the final segment dicts (facts from tree, language from LLM) ──
    segments_out: list[dict] = []
    for fact, llm in zip(interesting, llm_segments):
        size_pct = round(fact["samples"] / n_total * 100, 1)
        segments_out.append({
            "name": llm.get("name") or _rule_to_human(fact["rule"])[:48],
            "rule": _rule_to_human(fact["rule"]),
            "size": fact["samples"],
            "size_pct": size_pct,
            "avg_churn_prob": round(fact["value"], 3),
            "churn_rate": round(fact["value"], 3),   # for tree surrogate these are the same
            "characteristics": llm.get("characteristics") or [],
            "narrative": llm.get("narrative") or "",
            "recommended_actions": llm.get("recommended_actions") or [],
        })

    msg = f"Discovered {len(segments_out)} interpretable segment(s)"
    if narration_error:
        msg += f" (LLM naming failed: {narration_error})"

    return {
        "segments": segments_out,
        "current_step": "segment_discovery",
        "progress_messages": state.get("progress_messages", []) + [msg],
    }
