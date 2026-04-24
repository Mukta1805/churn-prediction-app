"""LangGraph StateGraph — wires the churn prediction pipeline."""

from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.prepare_target import prepare_target_node
from agents.horizon import horizon_definition_node
from agents.class_imbalance import class_imbalance_node
from agents.missing_values import missing_values_node
from agents.model_selection import (
    clean_data_node,
    run_model_pipeline_node,
    compute_shap_node,
)
from agents.business_aggregates import business_aggregates_node
from agents.segment_discovery import segment_discovery_node
from agents.insight_generation import generate_insights_node


def _route_after_target_prep(state: PipelineState) -> str:
    """Skip horizon definition entirely if no tenure column was provided."""
    tenure_col = state.get("schema", {}).get("tenure_col")
    return "horizon_definition" if tenure_col else "class_imbalance"


def build_graph():
    """Build and compile the churn prediction pipeline graph.

    Flow:
        prepare_target          (always)
        ── if schema.tenure_col present ──
            -> horizon_definition
        -> class_imbalance
        -> missing_values
        -> clean_data
        -> run_model_pipeline
        -> compute_shap
        -> business_aggregates  (BA-friendly numbers)
        -> segment_discovery    (tree surrogate + LLM naming)
        -> generate_insights    (structured + markdown)
        -> END
    """
    graph = StateGraph(PipelineState)

    graph.add_node("prepare_target", prepare_target_node)
    graph.add_node("horizon_definition", horizon_definition_node)
    graph.add_node("class_imbalance", class_imbalance_node)
    graph.add_node("missing_values", missing_values_node)
    graph.add_node("clean_data", clean_data_node)
    graph.add_node("run_model_pipeline", run_model_pipeline_node)
    graph.add_node("compute_shap", compute_shap_node)
    graph.add_node("business_aggregates", business_aggregates_node)
    graph.add_node("segment_discovery", segment_discovery_node)
    graph.add_node("generate_insights", generate_insights_node)

    graph.set_entry_point("prepare_target")
    graph.add_conditional_edges(
        "prepare_target",
        _route_after_target_prep,
        {
            "horizon_definition": "horizon_definition",
            "class_imbalance": "class_imbalance",
        },
    )
    graph.add_edge("horizon_definition", "class_imbalance")
    graph.add_edge("class_imbalance", "missing_values")
    graph.add_edge("missing_values", "clean_data")
    graph.add_edge("clean_data", "run_model_pipeline")
    graph.add_edge("run_model_pipeline", "compute_shap")
    graph.add_edge("compute_shap", "business_aggregates")
    graph.add_edge("business_aggregates", "segment_discovery")
    graph.add_edge("segment_discovery", "generate_insights")
    graph.add_edge("generate_insights", END)

    return graph.compile()
