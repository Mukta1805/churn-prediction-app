"""Streamlit UI for the Churn Prediction Multi-Agent System."""

import sys
import os

# Ensure project root is on the path so imports work with `streamlit run app.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from dotenv import load_dotenv

load_dotenv()

# Support Streamlit Cloud secrets (fallback to .env for local dev)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from agents.graph import build_graph
from agents.insight_generation import handle_chat_question
from agents.chart_agent import (
    pr_curve_figure,
    probability_distribution_figure,
    cumulative_gains_figure,
    lift_chart_figure,
)
from agents.simulation_agent import simulate_profit, explain_simulation
from agents.schema_detection import detect_schema
from pipeline.config import MODEL_DISPLAY_NAMES, BUSINESS_CONSTANTS, DEFAULT_SCHEMA

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("AI Churn Prediction Tool")
st.caption("LangGraph Multi-Agent System · d6tflow Pipeline · SHAP Explainability")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "sim_explanation" not in st.session_state:
    st.session_state.sim_explanation = None
if "sim_constants" not in st.session_state:
    st.session_state.sim_constants = None

# ---------------------------------------------------------------------------
# Sidebar — Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload any churn-style customer dataset. The schema is auto-detected — review it below before running.",
    )

    schema = None  # filled in after upload
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns")
        st.dataframe(raw_df.head(), height=200)

        # ── Schema detection + override ──
        st.subheader("Detected Schema")
        st.caption("Auto-detected from column names and dtypes. Override anything that's wrong.")

        detected = detect_schema(raw_df)
        all_cols = list(raw_df.columns)

        target_options = all_cols
        target_default = (
            target_options.index(detected["target_col"])
            if detected["target_col"] in target_options
            else 0
        )
        target_col = st.selectbox(
            "Target column (binary churn label)",
            options=target_options,
            index=target_default,
            help="The column the model will learn to predict. Must be binary.",
        )

        id_cols = st.multiselect(
            "ID / drop columns",
            options=all_cols,
            default=[c for c in detected["id_cols"] if c in all_cols],
            help="Columns dropped before modelling — IDs, free-text fields, etc.",
        )

        tenure_options = ["(none — skip horizon agent)"] + all_cols
        tenure_default = (
            tenure_options.index(detected["tenure_col"])
            if detected["tenure_col"] in tenure_options
            else 0
        )
        tenure_choice = st.selectbox(
            "Tenure column (months active)",
            options=tenure_options,
            index=tenure_default,
            help="Numeric column for customer age in months. If absent, horizon labelling is skipped.",
        )
        tenure_col = None if tenure_choice == tenure_options[0] else tenure_choice

        # ── Positive label ──
        unique_target_vals = (
            sorted(raw_df[target_col].dropna().unique().tolist(), key=str)
            if target_col in raw_df.columns
            else []
        )
        if len(unique_target_vals) == 2:
            pos_default = (
                detected["positive_label"]
                if detected["positive_label"] in unique_target_vals
                else unique_target_vals[-1]
            )
            positive_label = st.selectbox(
                "Positive label (= 'churned')",
                options=unique_target_vals,
                index=unique_target_vals.index(pos_default),
                help="Which raw value in the target column means the customer churned.",
            )
        else:
            positive_label = detected["positive_label"]

        schema = {
            "target_col": target_col,
            "id_cols": id_cols,
            "tenure_col": tenure_col,
            "positive_label": positive_label,
        }

        # ── Horizon (only meaningful if tenure_col is set) ──
        if tenure_col:
            st.subheader("Churn Horizon")
            selected_horizon = st.selectbox(
                "Define churn window",
                options=[30, 60, 90],
                format_func=lambda h: f"{h}-day churn",
                help="Customers who churned within this window will be labelled as churned (1).",
            )
        else:
            selected_horizon = 30  # ignored when horizon agent is skipped
            st.info("No tenure column — horizon labelling will be skipped.")

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    else:
        raw_df = None
        selected_horizon = 30
        run_btn = False

    st.divider()
    st.caption("Built with LangGraph + d6tflow + Streamlit")

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
if run_btn and raw_df is not None:
    # Reset previous results
    st.session_state.analysis_complete = False
    st.session_state.chat_history = []

    graph = build_graph()
    initial_state = {
        "raw_df": raw_df,
        "selected_horizon": selected_horizon,
        # Schema is auto-detected and (optionally) edited in the sidebar above.
        # Falls back to DEFAULT_SCHEMA if the upload block didn't produce one.
        "schema": schema if schema else dict(DEFAULT_SCHEMA),
    }

    has_horizon = bool(initial_state["schema"].get("tenure_col"))
    # 3 new nodes added: business_aggregates, segment_discovery, generate_insights (now 2 LLM calls)
    n_steps = 10 if has_horizon else 9
    offset = 1 if has_horizon else 0
    step_labels = {
        "prepare_target":       f"Step 1/{n_steps}: Encoding target column...",
        "horizon_definition":   f"Step 2/{n_steps}: Defining {selected_horizon}-day churn horizon...",
        "class_imbalance":      f"Step {2 + offset}/{n_steps}: Checking class imbalance...",
        "missing_values":       f"Step {3 + offset}/{n_steps}: Profiling and imputing missing values...",
        "clean_data":           f"Step {4 + offset}/{n_steps}: Cleaning data...",
        "run_model_pipeline":   f"Step {5 + offset}/{n_steps}: Training models with Bayesian optimization (~1-2 min)...",
        "compute_shap":         f"Step {6 + offset}/{n_steps}: Computing SHAP explanations...",
        "business_aggregates":  f"Step {7 + offset}/{n_steps}: Computing business aggregates (at-risk counts, revenue impact)...",
        "segment_discovery":    f"Step {8 + offset}/{n_steps}: Discovering customer segments...",
        "generate_insights":    f"Step {9 + offset}/{n_steps}: Generating business insights with AI...",
    }

    final_state = dict(initial_state)

    with st.status("Running churn analysis pipeline...", expanded=True) as status:
        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            final_state.update(node_output)

            label = step_labels.get(node_name, node_name)
            st.write(f"✓ {label.replace('...', ' — done!')}")

            # Show progress messages
            for msg in node_output.get("progress_messages", []):
                if msg not in (final_state.get("_shown_msgs") or []):
                    st.caption(f"  {msg}")

        status.update(label="Analysis complete!", state="complete", expanded=False)

    st.session_state.pipeline_state = final_state
    st.session_state.analysis_complete = True

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.analysis_complete:
    state = st.session_state.pipeline_state
    structured = state.get("structured_insights") or {}
    aggregates = state.get("business_aggregates") or {}
    segments = state.get("segments") or []
    summary = state.get("dataset_summary") or {}
    best = state.get("best_model_metrics") or {}

    (
        tab_exec,
        tab_risk,
        tab_why,
        tab_sim,
        tab_chat,
        tab_tech,
    ) = st.tabs([
        "🎯 Executive Summary",
        "👥 Who's at Risk?",
        "🔍 Why?",
        "🔬 Simulation",
        "💬 Ask Questions",
        "🔧 Technical Details",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Executive Summary  (the BA lands here first)
    # ══════════════════════════════════════════════════════════════════════
    with tab_exec:
        # Executive summary paragraph
        st.subheader("The Bottom Line")
        exec_summary = structured.get("executive_summary", "")
        if exec_summary:
            st.markdown(
                f"""<div style="font-size: 1.05rem; padding: 1rem 1.25rem;
                        background: rgba(37, 99, 235, 0.06);
                        border-left: 4px solid #2563EB; border-radius: 6px;
                        line-height: 1.6;">{exec_summary}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No executive summary generated.")

        st.markdown("")  # spacing

        # Hero KPI cards (computed from business_aggregates, always available)
        st.subheader("Key Numbers")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Customers Analysed", f"{summary.get('rows', 0):,}")
        k2.metric(
            "Observed Churn Rate",
            f"{summary.get('churn_rate_pct', 0):.2f}%",
            delta="historical",
            delta_color="off",
        )
        k3.metric(
            "Predicted At-Risk",
            f"{aggregates.get('at_risk_count', 0):,}",
            delta=f"{aggregates.get('at_risk_pct', 0)}% of test set",
            delta_color="off",
        )
        k4.metric(
            "Revenue at Stake",
            f"${aggregates.get('revenue_at_stake', 0):,.0f}",
            delta=f"@ ${aggregates.get('customer_value', 0)}/cust",
            delta_color="off",
        )
        k5.metric(
            "Retention Profit",
            f"${aggregates.get('projected_profit', 0):,.0f}",
            delta=f"threshold {aggregates.get('threshold_used', 0)}",
            delta_color="off",
        )

        st.divider()

        # Top 3 actions
        st.subheader("🎯 Your Top 3 Actions This Week")
        actions = structured.get("top_actions") or []
        if not actions:
            st.info("No action recommendations generated.")
        else:
            for i, action in enumerate(actions[:3], 1):
                effort = action.get("effort", "—")
                effort_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(effort, "⚪")
                with st.container(border=True):
                    a1, a2 = st.columns([2, 1])
                    with a1:
                        st.markdown(f"### {i}. {action.get('title', '')}")
                        st.write(action.get("description", ""))
                    with a2:
                        st.caption("**Impact**")
                        st.markdown(
                            f"<div style='font-size: 1rem; font-weight: 600; "
                            f"line-height: 1.4; margin-bottom: 0.5rem;'>"
                            f"{action.get('expected_impact', '—')}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{effort_icon} Effort: **{effort}**")
                        st.caption(f"⏱️ Timeline: **{action.get('timeline', '—')}**")

        st.divider()

        # Risk bucket pie chart
        st.subheader("Customer Base by Risk Level")
        bucket_counts = aggregates.get("risk_bucket_counts") or {}
        if bucket_counts:
            pc1, pc2 = st.columns([1, 1])
            with pc1:
                fig, ax = plt.subplots(figsize=(6, 6))
                labels = ["High risk\n(≥60%)", "Medium risk\n(30–60%)", "Low risk\n(<30%)"]
                sizes = [
                    bucket_counts.get("high", 0),
                    bucket_counts.get("medium", 0),
                    bucket_counts.get("low", 0),
                ]
                colors = ["#EF4444", "#F59E0B", "#10B981"]
                if sum(sizes) > 0:
                    wedges, texts, autotexts = ax.pie(
                        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                        startangle=90,
                        pctdistance=0.80,
                        labeldistance=1.12,
                        wedgeprops=dict(width=0.40, edgecolor="white"),
                        textprops={"fontsize": 11},
                    )
                    for autotext in autotexts:
                        autotext.set_color("white")
                        autotext.set_fontweight("bold")
                        autotext.set_fontsize(11)
                    ax.set_title("Predicted Churn Probability Buckets", pad=20)
                ax.set_aspect("equal")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with pc2:
                st.markdown("")
                st.markdown(
                    f"**🔴 High risk:** {bucket_counts.get('high', 0):,} customers "
                    f"(predicted ≥60% churn probability)"
                )
                st.markdown(
                    f"**🟡 Medium risk:** {bucket_counts.get('medium', 0):,} customers "
                    f"(30–60%)"
                )
                st.markdown(
                    f"**🟢 Low risk:** {bucket_counts.get('low', 0):,} customers "
                    f"(<30%)"
                )
                st.caption(
                    f"Based on {aggregates.get('test_set_size', 0):,} held-out "
                    "test-set customers. Scale to your full customer base accordingly."
                )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Who's at Risk?
    # ══════════════════════════════════════════════════════════════════════
    with tab_risk:
        st.subheader("Customer Segments at Risk")
        st.caption(
            "Segments discovered by training a small decision tree on the "
            "model's predictions. Rules are deterministic; names and "
            "recommendations are LLM-generated from those rules."
        )

        if not segments:
            st.info("No interpretable segments were discovered.")
        else:
            for seg in segments:
                p = seg.get("avg_churn_prob", 0)
                risk_badge = (
                    "🔴 High Risk" if p >= 0.6
                    else ("🟡 Medium Risk" if p >= 0.3 else "🟢 Low Risk")
                )
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"### {seg.get('name', 'Segment')}")
                        st.caption(risk_badge)
                    with c2:
                        st.metric("Customers", f"{seg.get('size', 0):,}")
                        st.caption(f"{seg.get('size_pct', 0)}% of base")
                    with c3:
                        st.metric("Predicted Churn", f"{p:.0%}")

                    if seg.get("narrative"):
                        st.markdown(f"_{seg['narrative']}_")

                    cc1, cc2 = st.columns([1, 1])
                    with cc1:
                        st.markdown("**Who they are:**")
                        for ch in seg.get("characteristics", []):
                            st.markdown(f"- {ch}")
                    with cc2:
                        st.markdown("**Recommended plays:**")
                        for ac in seg.get("recommended_actions", []):
                            st.markdown(f"- {ac}")

                    with st.expander("View the underlying rule"):
                        st.code(seg.get("rule", ""), language="text")

            # Comparison charts
            st.divider()
            st.subheader("Segment Comparison")
            sc1, sc2 = st.columns(2)
            names = [s.get("name", "?")[:22] for s in segments]
            with sc1:
                st.markdown("**Customer Count by Segment**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sizes = [s.get("size", 0) for s in segments]
                ax.barh(names, sizes, color="#2563EB")
                ax.set_xlabel("Customer Count")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with sc2:
                st.markdown("**Predicted Churn Rate by Segment**")
                fig, ax = plt.subplots(figsize=(6, 4))
                probs = [s.get("avg_churn_prob", 0) * 100 for s in segments]
                colors = [
                    "#EF4444" if p >= 60 else ("#F59E0B" if p >= 30 else "#10B981")
                    for p in probs
                ]
                ax.barh(names, probs, color=colors)
                ax.set_xlabel("Predicted Churn %")
                ax.set_xlim(0, 100)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Top at-risk individual customers
        st.divider()
        top_at_risk = aggregates.get("top_at_risk_customers") or []
        if top_at_risk:
            st.subheader("Top Individual Customers by Churn Probability")
            st.caption(
                "The 50 test-set customers with the highest predicted churn "
                "probability. Sort, filter, or download as a retention list."
            )
            top_df = pd.DataFrame(top_at_risk)
            if "_churn_probability" in top_df.columns:
                top_df["_churn_probability"] = (top_df["_churn_probability"] * 100).round(1)
                top_df = top_df.rename(columns={
                    "_churn_probability": "Churn Prob. (%)",
                    "_actual_churn": "Actual Churn",
                })
                # Reorder so the probability column is first
                cols = ["Churn Prob. (%)", "Actual Churn"] + [
                    c for c in top_df.columns
                    if c not in ("Churn Prob. (%)", "Actual Churn")
                ]
                top_df = top_df[cols]
            st.dataframe(top_df, use_container_width=True, hide_index=True, height=420)

    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — Why?
    # ══════════════════════════════════════════════════════════════════════
    with tab_why:
        st.subheader("What's Driving Churn")
        st.caption(
            "The factors most strongly associated with churn, translated from "
            "the model's feature analysis into business language."
        )

        drivers = structured.get("driver_narratives") or []
        if not drivers:
            st.info("No driver narratives generated.")
        else:
            for i, d in enumerate(drivers, 1):
                with st.container(border=True):
                    st.markdown(f"### {i}. {d.get('driver', 'Driver')}")
                    st.write(d.get("narrative", ""))
                    action = d.get("suggested_action", "")
                    if action:
                        st.markdown(f"💡 **What to do:** {action}")

        st.divider()
        st.subheader("Overall Feature Importance (Technical)")
        st.caption(
            "Raw feature importance from the model. The narratives above are "
            "the BA-friendly translation of these rankings."
        )
        importances = state.get("feature_importances") or []
        if importances:
            top_n = importances[:15]
            fig, ax = plt.subplots(figsize=(8, 5))
            fnames = [f["feature"] for f in reversed(top_n)]
            values = [f["importance"] for f in reversed(top_n)]
            ax.barh(fnames, values, color="#7C3AED")
            ax.set_xlabel("Mean |SHAP value|")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 3: Simulation ──
    with tab_sim:
        st.subheader("What-If Business Simulation")
        st.caption(
            "Adjust business assumptions and re-run the profit optimisation. "
            "No retraining — uses the trained model's test-set predictions."
        )

        baseline_metrics = state.get("best_model_metrics", {})
        predictions = state.get("predictions", {})

        @st.fragment
        def _simulation_fragment():
            """Scoped rerun — pressing Run Simulation only reruns this block,
            so the active tab never resets back to Executive Summary."""
            col_l, col_r = st.columns([1, 1])

            with col_l:
                st.markdown("**Adjust assumptions**")
                cv = st.slider(
                    "Customer lifetime value ($)",
                    min_value=100, max_value=2000,
                    value=BUSINESS_CONSTANTS["customer_value"], step=50,
                )
                cc = st.slider(
                    "Contact cost ($)",
                    min_value=1, max_value=100,
                    value=BUSINESS_CONSTANTS["contact_cost"], step=1,
                )
                rsr = st.slider(
                    "Retention success rate (%)",
                    min_value=5, max_value=80,
                    value=int(BUSINESS_CONSTANTS["retention_success_rate"] * 100), step=5,
                )
                mcl = st.slider(
                    "Missed churn loss ($)",
                    min_value=100, max_value=2000,
                    value=BUSINESS_CONSTANTS["missed_churn_loss"], step=50,
                )
                run_sim = st.button(
                    "Run Simulation", type="primary", use_container_width=True,
                    key="sim_run_btn",
                )

            if run_sim and predictions:
                new_constants = {
                    "customer_value": cv,
                    "contact_cost": cc,
                    "retention_success_rate": rsr / 100,
                    "missed_churn_loss": mcl,
                }
                with st.spinner("Computing..."):
                    result = simulate_profit(
                        y_test=predictions["y_test"],
                        y_prob=predictions["y_prob"],
                        **new_constants,
                    )
                    explanation = explain_simulation(
                        baseline_metrics=baseline_metrics,
                        baseline_constants=BUSINESS_CONSTANTS,
                        new_result=result,
                        new_constants=new_constants,
                    )
                st.session_state.sim_result = result
                st.session_state.sim_explanation = explanation
                st.session_state.sim_constants = new_constants

            with col_r:
                st.markdown("**Results**")
                sim = st.session_state.sim_result
                baseline_profit = baseline_metrics.get("expected_profit", 0)
                baseline_threshold = baseline_metrics.get("optimal_threshold", 0.5)

                if sim:
                    delta_profit = sim["expected_profit"] - baseline_profit
                    delta_threshold = sim["optimal_threshold"] - baseline_threshold

                    m1, m2 = st.columns(2)
                    m1.metric(
                        "Optimal Threshold",
                        f"{sim['optimal_threshold']:.3f}",
                        delta=f"{delta_threshold:+.3f}",
                    )
                    m2.metric(
                        "Expected Profit",
                        f"${sim['expected_profit']:,.0f}",
                        delta=f"${delta_profit:+,.0f}",
                    )
                    m3, m4 = st.columns(2)
                    m3.metric("Contacts Made", f"{sim['contacts_made']:,}")
                    m4.metric("Churners Missed", f"{sim['churners_missed']:,}")

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.plot(
                        baseline_metrics["threshold_curve"],
                        baseline_metrics["profit_curve"],
                        color="#C8D0E0", lw=1.5, label="Baseline",
                    )
                    ax.plot(
                        sim["threshold_curve"],
                        sim["profit_curve"],
                        color="#2563EB", lw=2, label="Simulation",
                    )
                    ax.axvline(baseline_threshold, color="#C8D0E0", linestyle="--", lw=1)
                    ax.axvline(sim["optimal_threshold"], color="#2563EB", linestyle="--", lw=1)
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel("Expected Profit ($)")
                    ax.set_title("Profit Curve: Baseline vs Simulation")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    if st.session_state.sim_explanation:
                        st.info(st.session_state.sim_explanation)
                else:
                    st.markdown(
                        f"Baseline — threshold: **{baseline_threshold:.3f}** · "
                        f"profit: **${baseline_profit:,.0f}**"
                    )
                    st.caption("Adjust the sliders and click Run Simulation to see what changes.")

        _simulation_fragment()

    # ══════════════════════════════════════════════════════════════════════
    # Tab 5 — Ask Questions (chat)
    # ══════════════════════════════════════════════════════════════════════
    with tab_chat:
        st.subheader("Ask Questions About the Analysis")
        st.caption(
            "Ask anything about the drivers, segments, numbers, or recommendations. "
            "The assistant has access to the full model results and segment data."
        )

        # Suggested questions
        st.markdown("**Suggested questions:**")
        sq_cols = st.columns(3)
        suggested_qs = [
            "Which segment should I prioritise and why?",
            "What's the ROI of a retention campaign?",
            "Which drivers are easiest to influence?",
        ]
        queued_question = None
        for i, q in enumerate(suggested_qs):
            if sq_cols[i].button(q, use_container_width=True, key=f"sq_{i}"):
                queued_question = q
        st.markdown("")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input (typed or queued from suggestion buttons)
        user_prompt = st.chat_input("Ask about the churn analysis...")
        if queued_question and not user_prompt:
            user_prompt = queued_question

        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_state = dict(state)
                    chat_state["chat_history"] = st.session_state.chat_history[:-1]
                    answer = handle_chat_question(chat_state, user_prompt)
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # ══════════════════════════════════════════════════════════════════════
    # Tab 6 — Technical Details (collapsible, data-science audit surface)
    # ══════════════════════════════════════════════════════════════════════
    with tab_tech:
        st.caption(
            "All the data-science content — model comparison, SHAP plots, PR curves, "
            "data profile. Open any section to audit the pipeline."
        )

        # ── Data Profile ──
        with st.expander("🔍 Data Profile — class imbalance & missing-value reasoning", expanded=False):
            st.markdown("**Class Imbalance Check**")
            imb = state.get("imbalance_config", {})
            if imb:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Minority class", f"{imb['minority_ratio']:.1%}")
                col_b.metric("Positives (churn=1)", f"{imb['minority_count']:,}")
                col_c.metric("Negatives (churn=0)", f"{imb['majority_count']:,}")
                if imb.get("is_imbalanced"):
                    st.warning(
                        f"**Imbalance detected.** Mitigation applied: "
                        f"`class_weight=balanced`, "
                        f"`scale_pos_weight={imb['xgb_scale_pos_weight']}`, "
                        f"CV metric=`{imb['primary_metric']}`."
                    )
                else:
                    st.success("Classes are balanced — no mitigation needed.")

            st.markdown("**Missing Values — LLM Reasoning**")
            strategies = state.get("missing_strategies", [])
            if not strategies:
                st.success("No missing values found in the dataset.")
            else:
                for s in strategies:
                    with st.expander(
                        f"**{s['column']}** — {s['missing_rate']:.1%} missing · "
                        f"strategy: `{s['strategy']}`"
                        + (f" → `{s['fill_value']}`" if s.get("fill_value") is not None else ""),
                        expanded=False,
                    ):
                        st.markdown(f"**Interpretation:** {s.get('interpretation', '—')}")
                        st.markdown(f"**Reasoning:** {s.get('reasoning', '—')}")

        # ── Model Results ──
        with st.expander("📊 Model Results — comparison, ROC-AUC, profit curve, SHAP", expanded=False):
            comparison = state.get("model_comparison", [])
            if comparison:
                comp_df = pd.DataFrame(comparison)
                display_df = comp_df[[
                    "display_name", "roc_auc", "pr_auc", "f1",
                    "runtime_sec", "optimal_threshold", "expected_profit"
                ]].copy()
                display_df.columns = [
                    "Model", "ROC-AUC", "PR-AUC", "F1",
                    "Runtime (s)", "Optimal Threshold", "Expected Profit ($)"
                ]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            if best:
                st.success(
                    f"**Best Model: {MODEL_DISPLAY_NAMES.get(best['model'], best['model'])}** — "
                    f"ROC-AUC: {best['roc_auc']} | "
                    f"Expected Profit: ${best['expected_profit']:,.0f}"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**ROC-AUC Comparison**")
                if comparison:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    names = [c["display_name"] for c in comparison]
                    aucs = [c["roc_auc"] for c in comparison]
                    colors = [
                        "#2563EB" if c["model"] == state.get("best_model_name") else "#C8D0E0"
                        for c in comparison
                    ]
                    ax.barh(names, aucs, color=colors)
                    ax.set_xlim(0.6, 1.0)
                    ax.set_xlabel("ROC-AUC")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            with col2:
                st.markdown("**Profit vs Threshold**")
                if best and "threshold_curve" in best:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(best["threshold_curve"], best["profit_curve"], color="#2563EB")
                    ax.axvline(
                        best["optimal_threshold"], color="red",
                        linestyle="--", label=f"Optimal: {best['optimal_threshold']:.3f}",
                    )
                    ax.set_xlabel("Classification Threshold")
                    ax.set_ylabel("Expected Profit ($)")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            # SHAP summary plot
            shap_vals = state.get("shap_values")
            feature_names = state.get("feature_names")
            if shap_vals is not None and feature_names is not None:
                st.markdown("**SHAP Summary Plot**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_vals, feature_names=feature_names,
                    show=False, max_display=20,
                )
                st.pyplot(plt.gcf())
                plt.close("all")

        # ── Diagnostic Charts ──
        with st.expander("📈 Diagnostic Charts — PR curve, gains, lift", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Precision-Recall Curve**")
                fig = pr_curve_figure(state)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            with col2:
                st.markdown("**Churn Probability Distribution**")
                fig = probability_distribution_figure(state)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Cumulative Gains**")
                fig = cumulative_gains_figure(state)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            with col4:
                st.markdown("**Lift Chart by Decile**")
                fig = lift_chart_figure(state)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

        # ── Legacy narrative ──
        with st.expander("💡 Full narrative insights (markdown)", expanded=False):
            legacy = state.get("auto_insights", "")
            if legacy:
                st.markdown(legacy)
            else:
                st.info("No narrative insights generated.")
else:
    # Landing page
    st.info("👈 Upload a CSV dataset in the sidebar to get started.")

    st.markdown("### Built for retention analysts, not data scientists")
    st.markdown(
        "Upload any customer dataset with a churn label. The tool handles the ML "
        "behind the scenes and surfaces the things you actually need to act on:"
    )

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("#### 🎯 Executive Summary")
        st.caption(
            "Bottom-line KPIs — at-risk count, revenue at stake, projected retention "
            "profit, and the top 3 things you should do this week."
        )
    with lc2:
        st.markdown("#### 👥 Customer Segments")
        st.caption(
            "Auto-discovered groups with shared churn drivers. Each segment comes "
            "with a narrative explanation and a recommended retention playbook."
        )
    with lc3:
        st.markdown("#### 🔍 Root Causes")
        st.caption(
            "Plain-English explanations of what's driving churn and what you can "
            "do about each driver — no SHAP values or ROC curves required."
        )

    with st.expander("What happens when I click Run Analysis?"):
        st.markdown("""
The system runs an automated multi-agent pipeline (hidden from you by default):

1. **Schema detection** — auto-detects target, ID, and tenure columns from your CSV
2. **Target encoding** — handles Yes/No, True/False, 1/0, etc.
3. **Horizon definition** *(if tenure exists)* — builds 30/60/90-day churn labels
4. **Class imbalance** — detects rare-event skew and applies mitigation
5. **Missing values** — LLM reasons about each column and auto-imputes
6. **Data cleaning** — drops ID columns and computes summary stats
7. **Model training** — 5 ML models with Bayesian hyperparameter tuning
8. **SHAP explainability** — extracts feature-level importance
9. **Business aggregates** — computes at-risk counts, revenue at stake, risk buckets
10. **Segment discovery** — surrogate decision tree + LLM naming for interpretable segments
11. **Insight generation** — AI produces the executive summary, top actions, and driver narratives

All of this takes ~90 seconds. Technical details (model comparison, SHAP plots, PR curves)
are still available under the **Technical Details** tab if you or a DS colleague want to audit.
        """)
