"""Agent — Results Explainer: interpret model comparison & charts + interactive Q&A."""

from openai import OpenAI

from agents.state import PipelineState
from pipeline.config import MODEL_DISPLAY_NAMES


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _format_model_table(state: PipelineState) -> str:
    comparison = state.get("model_comparison", []) or []
    if not comparison:
        return "No model comparison results available."

    lines = ["Model | ROC-AUC | PR-AUC | F1 | Threshold | Expected Profit", "--- | --- | --- | --- | --- | ---"]
    for m in comparison:
        lines.append(
            f"{m.get('display_name', m.get('model', 'N/A'))} | {m.get('roc_auc', '—')} | {m.get('pr_auc', '—')} | "
            f"{m.get('f1', '—')} | {float(m.get('optimal_threshold', 0.0)):.3f} | ${float(m.get('expected_profit', 0.0)):,.0f}"
        )
    return "\n".join(lines)


def _build_results_context(state: PipelineState) -> str:
    project_overview = (state.get("project_overview") or "").strip()
    summary = state.get("dataset_summary", {}) or {}
    best = state.get("best_model_metrics", {}) or {}
    best_model = MODEL_DISPLAY_NAMES.get(best.get("model", ""), best.get("model", "N/A"))
    top_feats = (state.get("feature_importances", []) or [])[:10]

    feat_lines = []
    for i, f in enumerate(top_feats):
        feat_lines.append(f"{i+1}. {f.get('feature', '—')} (mean |SHAP|: {f.get('importance', 0):.4f})")

    overview_block = ""
    if project_overview:
        overview_block = f"PROJECT / DATASET OVERVIEW (USER-PROVIDED):\n{project_overview}\n\n"

    return f"""
{overview_block}DATASET SUMMARY:
- Rows: {summary.get('rows', '—')}
- Columns: {summary.get('columns', '—')}
- Churn rate: {summary.get('churn_rate_pct', '—')}%

MODEL COMPARISON TABLE:
{_format_model_table(state)}

BEST MODEL:
- Model: {best_model}
- ROC-AUC: {best.get('roc_auc', '—')}
- PR-AUC: {best.get('pr_auc', '—')}
- F1: {best.get('f1', '—')}
- Optimal threshold: {best.get('optimal_threshold', '—')}
- Expected profit @ optimal threshold: ${best.get('expected_profit', 0):,.0f}

TOP SHAP FEATURES (global importance):
{chr(10).join(feat_lines) if feat_lines else '—'}

CHARTS AVAILABLE IN THE APP:
- ROC-AUC bar chart (Model Results tab)
- Profit vs Threshold (Model Results tab)
- Precision-Recall Curve (Charts tab)
- Churn Probability Distribution (Charts tab)
- Cumulative Gains (Charts tab)
- Lift Chart by Decile (Charts tab)
""".strip()


RESULTS_EXPLAINER_SYSTEM_PROMPT = """You are a senior analytics consultant helping a non-technical stakeholder understand churn model results.
You will be given:
- a user-provided project/dataset overview (business context)
- model comparison metrics (ROC-AUC, PR-AUC, F1), an optimal threshold and expected profit
- a list of top SHAP features (global importance)
- the names of charts shown in the UI

Write a clear, practical explanation in markdown that:
- defines each metric (ROC-AUC, PR-AUC, F1) in plain language and when it matters
- explains why a model might look good on one metric but not another (esp. class imbalance)
- explains how to read each chart listed, and what “good” vs “bad” looks like
- explains what the optimal threshold means operationally (trade-offs, who gets contacted)
- ties the interpretation back to the project overview (use the business context)

Be concise but concrete. Avoid math-heavy exposition. If a number is missing, state what you would need."""


def explain_results(state: PipelineState) -> str:
    """Generate a one-shot explanation for the Model Results + Charts."""
    client = _get_client()
    context = _build_results_context(state)
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": RESULTS_EXPLAINER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


RESULTS_QA_SYSTEM_PROMPT = """You are an analytics assistant answering questions about churn model results and charts.
Always incorporate the user-provided project/dataset overview when it helps interpret the results.
Be specific and refer back to the provided metrics, thresholds, profit framing, SHAP features, and chart meanings.
If a question requires data that is not present, say what is missing and suggest what to check next."""


def handle_results_question(state: PipelineState, question: str, chat_history: list[dict] | None = None) -> str:
    """Interactive Q&A grounded in the same results context."""
    client = _get_client()
    context = _build_results_context(state)
    history = chat_history or []

    messages: list[dict] = [
        {"role": "system", "content": RESULTS_QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the analysis context:\n\n{context}"},
    ]
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

