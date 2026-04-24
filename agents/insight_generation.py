"""Agent 2 — Insight Generation.

Produces TWO LLM outputs from the same model-results context:

  1. `structured_insights` (new) — JSON with executive_summary, kpis, top_actions,
     driver_narratives. Drives the Executive Summary / Why tabs in the UI.

  2. `auto_insights` (legacy) — long-form markdown. Kept for the Insights tab and
     as richer context for the chat agent.

Standalone `handle_chat_question` is unchanged in behaviour, but now has access
to segments and business aggregates in its context.
"""

from __future__ import annotations

import json

from openai import OpenAI

from agents.state import PipelineState
from pipeline.config import MODEL_DISPLAY_NAMES


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Build the context string used by all three LLM calls
# ---------------------------------------------------------------------------
def _build_context(state: PipelineState) -> str:
    project_overview = (state.get("project_overview") or "").strip()
    summary = state.get("dataset_summary", {})
    comparison = state.get("model_comparison", [])
    best = state.get("best_model_metrics", {})
    importances = state.get("feature_importances", [])
    aggregates = state.get("business_aggregates", {})
    segments = state.get("segments", [])

    # Model comparison table
    table_lines = ["Model | ROC-AUC | PR-AUC | F1 | Threshold | Expected Profit"]
    table_lines.append("--- | --- | --- | --- | --- | ---")
    for m in comparison:
        table_lines.append(
            f"{m['display_name']} | {m['roc_auc']} | {m['pr_auc']} | "
            f"{m['f1']} | {m['optimal_threshold']:.3f} | ${m['expected_profit']:,.0f}"
        )

    # Top features
    top_features = importances[:15]
    feat_lines = [f"  {i+1}. {f['feature']} (SHAP importance: {f['importance']:.4f})"
                  for i, f in enumerate(top_features)]

    # Business aggregates (new)
    agg_lines = []
    if aggregates:
        agg_lines = [
            f"- At-risk customers: {aggregates.get('at_risk_count', 0):,} "
            f"({aggregates.get('at_risk_pct', 0)}% of customer base)",
            f"- Revenue at stake: ${aggregates.get('revenue_at_stake', 0):,.0f}",
            f"- Projected profit at optimal threshold: ${aggregates.get('projected_profit', 0):,.0f}",
            f"- Risk bucket counts: {aggregates.get('risk_bucket_counts', {})}",
        ]

    # Segments (new)
    seg_lines = []
    if segments:
        for i, s in enumerate(segments[:6], 1):
            seg_lines.append(
                f"  {i}. {s.get('name', 'Unnamed segment')} — "
                f"{s.get('size', 0)} customers ({s.get('size_pct', 0)}%), "
                f"predicted churn rate {s.get('avg_churn_prob', 0):.1%}. "
                f"Rule: {s.get('rule', '')}"
            )

    overview_block = ""
    if project_overview:
        overview_block = (
            f"PROJECT / DATASET OVERVIEW (USER-PROVIDED):\n{project_overview}\n\n"
        )

    context = f"""
{overview_block}DATASET SUMMARY:
- {summary.get('rows', '?')} customers, {summary.get('columns', '?')} features
- Churn rate: {summary.get('churn_rate_pct', '?')}%
- {summary.get('rows_dropped', 0)} rows dropped due to missing values
- Numeric features: {', '.join(summary.get('numeric_features', [])[:10])}
- Categorical features: {', '.join(summary.get('categorical_features', []))}

BUSINESS AGGREGATES:
{chr(10).join(agg_lines) if agg_lines else '- (not computed)'}

CUSTOMER SEGMENTS (from decision-tree surrogate):
{chr(10).join(seg_lines) if seg_lines else '- (not discovered)'}

MODEL COMPARISON:
{chr(10).join(table_lines)}

BEST MODEL: {MODEL_DISPLAY_NAMES.get(best.get('model', ''), best.get('model', 'N/A'))}
- ROC-AUC: {best.get('roc_auc', 'N/A')}
- PR-AUC: {best.get('pr_auc', 'N/A')}
- F1 Score: {best.get('f1', 'N/A')}
- Optimal classification threshold: {best.get('optimal_threshold', 'N/A')}
- Expected profit at optimal threshold: ${best.get('expected_profit', 0):,.0f}
- Business assumptions: customer_value=$500, contact_cost=$10, retention_success_rate=25%, missed_churn_loss=$500

TOP FEATURES BY SHAP IMPORTANCE:
{chr(10).join(feat_lines)}
""".strip()

    return context


# ---------------------------------------------------------------------------
# Prompt 1 — structured JSON insights (drives Executive Summary + Why tabs)
# ---------------------------------------------------------------------------
_STRUCTURED_SYSTEM = """\
You are a retention strategy analyst producing structured output for a business dashboard.
You will receive churn-prediction analysis results — optionally prefixed with a user-provided
project/dataset overview describing the business context — and must respond with JSON in the
exact schema below. Every field is required. When the overview is present, tailor the
executive summary, actions, and driver narratives to that business context.

{
  "executive_summary": "2-3 sentence paragraph summarising the churn situation and what matters most. Business language only — no mention of ROC-AUC, SHAP, or model names.",
  "kpis": [
    {"label": "short KPI name", "value": "formatted value string", "context": "short explanation or comparison"}
    // produce 3-4 KPIs that matter to a retention manager
  ],
  "top_actions": [
    {"title": "Imperative-voice action title (5-8 words)",
     "description": "1-2 sentence description of the play",
     "expected_impact": "quantified when possible, e.g. 'Protect $12K in revenue'",
     "effort": "Low | Medium | High",
     "timeline": "e.g. 'This week', '2 weeks', '30 days'"}
    // produce exactly 3 prioritised actions
  ],
  "driver_narratives": [
    {"driver": "plain-English driver name (NOT the raw feature name)",
     "narrative": "1-2 sentence explanation of how this drives churn, using actual numbers from the data",
     "suggested_action": "1 sentence, what to do about it"}
    // produce 4-5 driver narratives, from most to least important
  ]
}

Rules:
- Reference actual numbers and feature names from the data provided.
- Executive summary must NOT contain technical terms.
- Driver names should be BA-friendly ("Inactive membership" not "IsActiveMember").
- Every action's expected_impact should tie to the actual at-risk count or revenue figures.
- Return ONLY the JSON object. No preamble."""


def _generate_structured_insights(context: str) -> dict:
    """Second LLM call — structured JSON suitable for rich UI rendering."""
    try:
        response = _get_client().chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": _STRUCTURED_SYSTEM},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "executive_summary": f"(Insight generation failed: {e})",
            "kpis": [],
            "top_actions": [],
            "driver_narratives": [],
        }


# ---------------------------------------------------------------------------
# Prompt 2 — legacy markdown insights (kept for Insights tab + chat context)
# ---------------------------------------------------------------------------
_MARKDOWN_SYSTEM = """You are a senior business analyst specializing in customer churn prediction.
You will receive ML model results and SHAP feature analysis from a churn prediction pipeline,
optionally prefixed with a user-provided project/dataset overview describing the business
context. Use the overview, when present, to frame your analysis.
Provide clear, actionable business insights in markdown format.

Structure your response as:
## Key Findings
- Top 5 churn risk drivers with business explanations

## At-Risk Customer Segments
- Describe 3-4 customer segments most at risk, using feature ranges

## Retention Recommendations
- 3-5 specific, actionable retention strategies tied to the data

## Business Impact
- Expected profit impact based on the threshold optimization
- What the optimal threshold means in practical terms

Be specific — reference actual feature names and values from the data. Avoid generic advice."""


def _generate_markdown_insights(context: str) -> str:
    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": _MARKDOWN_SYSTEM},
            {"role": "user", "content": context},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def generate_insights_node(state: PipelineState) -> dict:
    context = _build_context(state)

    structured = _generate_structured_insights(context)
    markdown = _generate_markdown_insights(context)

    return {
        "auto_insights": markdown,
        "structured_insights": structured,
        "chat_history": [],
        "current_step": "Insights generated",
        "progress_messages": state.get("progress_messages", []) + [
            "Generated structured + narrative business insights",
        ],
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
_CHAT_SYSTEM = """You are a churn-prediction analyst assistant. You have access to the results
of a machine learning pipeline, including model metrics, customer segments, business aggregates,
and feature importance analysis. You may also receive a user-provided project/dataset overview
describing the business context — incorporate it when answering.

Answer questions about the model results, feature importance, customer segments, business
impact, and retention strategies. Be specific and cite real numbers from the data provided.

If asked about something not covered by the data, say so clearly."""


def handle_chat_question(state: PipelineState, question: str) -> str:
    context = _build_context(state)
    chat_history = state.get("chat_history", [])

    messages = [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user", "content": f"Here are the analysis results:\n\n{context}"},
        {"role": "assistant", "content": "I have the full analysis results. What would you like to know?"},
    ]
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    response = _get_client().chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content
