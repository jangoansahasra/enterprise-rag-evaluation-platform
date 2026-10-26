"""Saved-results dashboard for enterprise RAG reliability experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_RESULTS = (
    ROOT
    / "artifacts"
    / "baselines"
    / "retrieval-comparison-v0.2.json"
)
WORKFLOW_RESULTS = (
    ROOT
    / "artifacts"
    / "baselines"
    / "workflow-decisions-v0.1.json"
)
ANSWER_QUALITY_RESULTS = (
    ROOT
    / "artifacts"
    / "baselines"
    / "answer-quality-v0.1.json"
)
QUALITY_GATES = ROOT / "config" / "quality-gates.json"
SOURCES = ROOT / "data" / "sources.json"
BENCHMARK = ROOT / "data" / "benchmark.json"


@st.cache_data
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_frame(results: dict) -> pd.DataFrame:
    display_names = {
        "bm25": "BM25",
        "dense": "Dense",
        "hybrid_rrf": "Hybrid RRF",
        "hybrid_reranked": "Hybrid + reranking",
    }

    rows = []

    for key, report in results.items():
        rows.append(
            {
                "configuration": display_names.get(key, key),
                "recall_at_3": report["summary"]["recall_at_3"],
                "mrr": report["summary"]["mean_reciprocal_rank"],
                "mean_latency_ms": report["timing"]["mean_query_time_ms"],
                "evaluated_questions": report["summary"][
                    "evaluated_questions"
                ],
            }
        )

    return pd.DataFrame(rows)


def retrieval_failures(results: dict) -> pd.DataFrame:
    display_names = {
        "bm25": "BM25",
        "dense": "Dense",
        "hybrid_rrf": "Hybrid RRF",
        "hybrid_reranked": "Hybrid + reranking",
    }

    rows = []

    for key, report in results.items():
        for question in report["questions"]:
            if question["first_relevant_rank"] is None:
                rows.append(
                    {
                        "configuration": display_names.get(key, key),
                        "question_id": question["question_id"],
                        "category": question["category"],
                        "expected_chunks": ", ".join(
                            question["relevant_chunk_ids"]
                        ),
                        "retrieved_chunks": ", ".join(
                            question["retrieved_chunk_ids"]
                        ),
                    }
                )

    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Enterprise RAG Reliability",
    page_icon="📊",
    layout="wide",
)

st.title("Enterprise RAG Reliability & Evaluation Platform")
st.caption(
    "Credential-free dashboard generated from versioned evaluation artifacts"
)

retrieval_results = load_json(RETRIEVAL_RESULTS)
workflow_results = load_json(WORKFLOW_RESULTS)
answer_quality_results = load_json(ANSWER_QUALITY_RESULTS)
quality_gates = load_json(QUALITY_GATES)
sources = load_json(SOURCES)
benchmark = load_json(BENCHMARK)

retrieval_df = retrieval_frame(retrieval_results)
failure_df = retrieval_failures(retrieval_results)

best_recall = retrieval_df["recall_at_3"].max()
best_mrr = retrieval_df["mrr"].max()
workflow_accuracy = workflow_results["summary"]["decision_accuracy"]
answer_quality_summary = answer_quality_results["summary"]

(
    overview_tab,
    retrieval_tab,
    answer_quality_tab,
    workflow_tab,
    readiness_tab,
) = st.tabs(
    [
        "Overview",
        "Retrieval experiments",
        "Answer quality",
        "Workflow decisions",
        "Production readiness",
    ]
)

with overview_tab:
    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Best Recall@3",
        f"{best_recall:.1%}",
    )
    metric_columns[1].metric(
        "Best MRR",
        f"{best_mrr:.4f}",
    )
    metric_columns[2].metric(
        "Decision accuracy",
        f"{workflow_accuracy:.1%}",
    )
    metric_columns[3].metric(
        "Verified benchmark",
        f"{len(benchmark)} questions",
    )

    st.subheader("Current evaluation scope")

    scope_columns = st.columns(3)
    scope_columns[0].metric("Public sources", len(sources))
    scope_columns[1].metric(
        "Retrieval configurations",
        len(retrieval_df),
    )
    scope_columns[2].metric(
        "Answerable retrieval cases",
        int(retrieval_df["evaluated_questions"].max()),
    )

    st.info(
        "The dashboard uses saved evaluation artifacts and does not require "
        "hosted-model credentials."
    )

    st.warning(
        "The current benchmark is a manually verified starter set. "
        "Its results demonstrate the evaluation framework but do not yet "
        "establish production readiness."
    )

with retrieval_tab:
    st.subheader("Retrieval quality comparison")

    quality_long = retrieval_df.melt(
        id_vars=["configuration"],
        value_vars=["recall_at_3", "mrr"],
        var_name="metric",
        value_name="score",
    )

    quality_chart = px.bar(
        quality_long,
        x="configuration",
        y="score",
        color="metric",
        barmode="group",
        range_y=[0, 1],
        labels={
            "configuration": "Configuration",
            "score": "Score",
            "metric": "Metric",
        },
        color_discrete_map={
            "recall_at_3": "#2563EB",
            "mrr": "#14B8A6",
        },
    )
    quality_chart.update_layout(
    legend_title_text="",
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
    },
    margin={"r": 40},
)
    
    st.plotly_chart(quality_chart, width="stretch")

    st.subheader("Quality and latency tradeoff")

    tradeoff_chart = px.scatter(
        retrieval_df,
        x="mean_latency_ms",
        y="mrr",
        size="recall_at_3",
        color="configuration",
        hover_data=["recall_at_3"],
        labels={
            "mean_latency_ms": "Mean query time (ms)",
            "mrr": "Mean reciprocal rank",
            "configuration": "Configuration",
        },
    )
    tradeoff_chart.update_layout(
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
    },
    margin={"r": 40},
)
    st.plotly_chart(tradeoff_chart, width="stretch")

    st.dataframe(
        retrieval_df.style.format(
            {
                "recall_at_3": "{:.2%}",
                "mrr": "{:.4f}",
                "mean_latency_ms": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Latency values come from one local run over a small corpus and are "
        "directional rather than production benchmarks."
    )
with answer_quality_tab:
    st.subheader("Grounding and answer-quality evaluation")

    primary_metrics = st.columns(4)
    primary_metrics[0].metric(
        "Answer correctness",
        f"{answer_quality_summary['answer_correctness']:.1%}",
    )
    primary_metrics[1].metric(
        "Faithfulness",
        f"{answer_quality_summary['faithfulness']:.1%}",
    )
    primary_metrics[2].metric(
        "Citation precision",
        f"{answer_quality_summary['citation_precision']:.1%}",
    )
    primary_metrics[3].metric(
        "Citation recall",
        f"{answer_quality_summary['citation_recall']:.1%}",
    )

    secondary_metrics = st.columns(4)
    secondary_metrics[0].metric(
        "Hallucination rate",
        f"{answer_quality_summary['hallucination_rate']:.1%}",
    )
    secondary_metrics[1].metric(
        "Decision accuracy",
        f"{answer_quality_summary['decision_accuracy']:.1%}",
    )
    secondary_metrics[2].metric(
        "Refusal accuracy",
        f"{answer_quality_summary['refusal_accuracy']:.1%}",
    )
    secondary_metrics[3].metric(
        "Escalation accuracy",
        f"{answer_quality_summary['escalation_accuracy']:.1%}",
    )

    chart_metrics = pd.DataFrame(
        [
            {
                "metric": "Answer correctness",
                "score": answer_quality_summary[
                    "answer_correctness"
                ],
            },
            {
                "metric": "Faithfulness",
                "score": answer_quality_summary["faithfulness"],
            },
            {
                "metric": "Citation precision",
                "score": answer_quality_summary[
                    "citation_precision"
                ],
            },
            {
                "metric": "Citation recall",
                "score": answer_quality_summary["citation_recall"],
            },
            {
                "metric": "Decision accuracy",
                "score": answer_quality_summary[
                    "decision_accuracy"
                ],
            },
        ]
    )

    answer_chart = px.bar(
        chart_metrics,
        x="metric",
        y="score",
        range_y=[0, 1],
        color="score",
        color_continuous_scale="RdYlGn",
        labels={
            "metric": "Metric",
            "score": "Score",
        },
    )
    answer_chart.update_layout(
        coloraxis_showscale=False,
        showlegend=False,
    )
    st.plotly_chart(answer_chart, width="stretch")

    st.caption(
        "Answer correctness uses normalized lexical token F1. "
        "Faithfulness measures answer-term coverage by retrieved evidence. "
        "These are deterministic proxy metrics rather than LLM judge scores."
    )

    answer_cases = pd.DataFrame(answer_quality_results["cases"])
    citation_issues = answer_cases[
        (
            answer_cases["citation_precision"].fillna(1.0)
            < 1.0
        )
        | (
            answer_cases["citation_recall"].fillna(1.0)
            < 1.0
        )
        | (~answer_cases["decision_correct"])
    ]

    st.subheader("Cases requiring review")

    if citation_issues.empty:
        st.success("No answer-quality issues were recorded.")
    else:
        st.dataframe(
            citation_issues[
                [
                    "question_id",
                    "question",
                    "answer",
                    "citation_precision",
                    "citation_recall",
                    "citations",
                    "expected_citations",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        st.warning(
            "The baseline identifies a wrong citation for q015 and "
            "incomplete multi-document citation coverage for q019."
        )

with workflow_tab:
    st.subheader("Answer, refusal, and escalation behavior")

    cases = pd.DataFrame(workflow_results["cases"])

    decision_counts = (
        cases.groupby("actual_decision", as_index=False)
        .size()
        .rename(columns={"size": "questions"})
    )

    decision_chart = px.bar(
        decision_counts,
        x="actual_decision",
        y="questions",
        color="actual_decision",
        labels={
            "actual_decision": "Decision",
            "questions": "Questions",
        },
        color_discrete_map={
            "answer": "#16A34A",
            "refuse": "#DC2626",
            "escalate": "#F59E0B",
        },
    )
    decision_chart.update_layout(showlegend=False)
    st.plotly_chart(decision_chart, width="stretch")

    selected_decision = st.selectbox(
        "Filter by decision",
        ["All", "answer", "refuse", "escalate"],
    )

    filtered_cases = cases
    if selected_decision != "All":
        filtered_cases = cases[
            cases["actual_decision"] == selected_decision
        ]

    st.dataframe(
        filtered_cases[
            [
                "question_id",
                "question",
                "expected_decision",
                "actual_decision",
                "classification",
                "reason",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

with readiness_tab:
    st.subheader("Production-readiness assessment")

    bm25_row = retrieval_df[
        retrieval_df["configuration"] == "BM25"
    ].iloc[0]

    gates = [
        {
            "gate": "Minimum public sources",
            "observed": len(sources),
            "required": quality_gates["dataset"]["minimum_sources"],
            "passed": (
                len(sources)
                >= quality_gates["dataset"]["minimum_sources"]
            ),
        },
        {
            "gate": "Minimum benchmark questions",
            "observed": len(benchmark),
            "required": quality_gates["dataset"][
                "minimum_benchmark_questions"
            ],
            "passed": (
                len(benchmark)
                >= quality_gates["dataset"][
                    "minimum_benchmark_questions"
                ]
            ),
        },
        {
            "gate": "BM25 Recall@3",
            "observed": bm25_row["recall_at_3"],
            "required": quality_gates["retrieval"][
                "minimum_recall_at_3"
            ],
            "passed": (
                bm25_row["recall_at_3"]
                >= quality_gates["retrieval"][
                    "minimum_recall_at_3"
                ]
            ),
        },
        {
            "gate": "BM25 MRR",
            "observed": bm25_row["mrr"],
            "required": quality_gates["retrieval"][
                "minimum_mean_reciprocal_rank"
            ],
            "passed": (
                bm25_row["mrr"]
                >= quality_gates["retrieval"][
                    "minimum_mean_reciprocal_rank"
                ]
            ),
        },
        {
            "gate": "Workflow decision accuracy",
            "observed": workflow_accuracy,
            "required": 0.95,
            "passed": workflow_accuracy >= 0.95,
        },
    ]

    gates_df = pd.DataFrame(gates)
    gates_df["status"] = gates_df["passed"].map(
        {True: "PASS", False: "FAIL"}
    )

    st.dataframe(
        gates_df[
            ["gate", "observed", "required", "status"]
        ],
        width="stretch",
        hide_index=True,
    )

    if gates_df["passed"].all():
        st.success("All current starter-set quality gates pass.")

    st.error(
        "Production status: NOT READY. Expand the benchmark, add held-out "
        "evaluation data, add semantic judge-based evaluation, and validate "
        "latency under realistic load before release."
    )

    st.subheader("Retrieval failures")

    if failure_df.empty:
        st.success("No retrieval failures were recorded.")
    else:
        st.dataframe(
            failure_df,
            width="stretch",
            hide_index=True,
        )