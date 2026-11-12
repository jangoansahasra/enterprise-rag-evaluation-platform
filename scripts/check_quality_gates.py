"""Apply deterministic CI quality gates to the live RAG workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from rag_reliability.answer_evaluation import (
    evaluate_answer,
    summarize_answer_evaluations,
)
from rag_reliability.data import (
    load_benchmark,
    load_corpus,
)
from rag_reliability.evaluation import evaluate_retriever
from rag_reliability.retrieval import BM25Retriever
from rag_reliability.workflow import build_workflow
from run_workflow_demo import build_retriever


ROOT = Path(__file__).resolve().parents[1]
Direction = Literal["minimum", "maximum"]


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def required_metric(
    summary: dict[str, Any],
    name: str,
) -> float:
    """Return a required numeric metric or fail clearly."""
    value = summary.get(name)

    if value is None:
        raise ValueError(
            f"Required metric is unavailable: {name}"
        )

    return float(value)


def evaluate_live_workflow(
    benchmark: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run every benchmark item through the current workflow once."""
    workflow = build_workflow(build_retriever())
    cases = []

    for item in benchmark:
        output = workflow.invoke(
            {"question": item.question}
        )
        cases.append(
            evaluate_answer(
                item,
                output,
            )
        )

    return summarize_answer_evaluations(cases), cases


def build_checks(
    observed: dict[str, float],
    thresholds: dict[str, Any],
) -> list[tuple[str, float, Direction, float]]:
    """Build minimum and maximum quality checks."""
    dataset = thresholds["dataset"]
    retrieval = thresholds["retrieval"]
    workflow = thresholds["workflow"]
    answer_quality = thresholds["answer_quality"]

    return [
        (
            "sources",
            observed["sources"],
            "minimum",
            float(dataset["minimum_sources"]),
        ),
        (
            "corpus_passages",
            observed["corpus_passages"],
            "minimum",
            float(dataset["minimum_corpus_passages"]),
        ),
        (
            "benchmark_questions",
            observed["benchmark_questions"],
            "minimum",
            float(dataset["minimum_benchmark_questions"]),
        ),
        (
            "recall_at_3",
            observed["recall_at_3"],
            "minimum",
            float(retrieval["minimum_recall_at_3"]),
        ),
        (
            "mean_reciprocal_rank",
            observed["mean_reciprocal_rank"],
            "minimum",
            float(
                retrieval[
                    "minimum_mean_reciprocal_rank"
                ]
            ),
        ),
        (
            "decision_accuracy",
            observed["decision_accuracy"],
            "minimum",
            float(
                workflow["minimum_decision_accuracy"]
            ),
        ),
        (
            "answer_correctness",
            observed["answer_correctness"],
            "minimum",
            float(
                answer_quality[
                    "minimum_answer_correctness"
                ]
            ),
        ),
        (
            "faithfulness",
            observed["faithfulness"],
            "minimum",
            float(
                answer_quality["minimum_faithfulness"]
            ),
        ),
        (
            "hallucination_rate",
            observed["hallucination_rate"],
            "maximum",
            float(
                answer_quality[
                    "maximum_hallucination_rate"
                ]
            ),
        ),
        (
            "citation_precision",
            observed["citation_precision"],
            "minimum",
            float(
                answer_quality[
                    "minimum_citation_precision"
                ]
            ),
        ),
        (
            "citation_recall",
            observed["citation_recall"],
            "minimum",
            float(
                answer_quality[
                    "minimum_citation_recall"
                ]
            ),
        ),
        (
            "refusal_accuracy",
            observed["refusal_accuracy"],
            "minimum",
            float(
                answer_quality[
                    "minimum_refusal_accuracy"
                ]
            ),
        ),
        (
            "escalation_accuracy",
            observed["escalation_accuracy"],
            "minimum",
            float(
                answer_quality[
                    "minimum_escalation_accuracy"
                ]
            ),
        ),
    ]


def check_passed(
    value: float,
    direction: Direction,
    threshold: float,
) -> bool:
    """Apply one directional threshold."""
    if direction == "minimum":
        return value >= threshold
    return value <= threshold


def main() -> None:
    """Evaluate current code and fail on quality regressions."""
    thresholds = load_json(
        ROOT / "config/quality-gates.json"
    )
    sources = load_json(
        ROOT / "data/sources.json"
    )
    benchmark = load_benchmark(
        ROOT / "data/benchmark.json"
    )
    corpus = load_corpus(
        ROOT / "data/corpus.json"
    )

    retrieval_report = evaluate_retriever(
        BM25Retriever(corpus),
        benchmark,
        top_k=3,
    )
    retrieval_summary = retrieval_report["summary"]

    print("Running live workflow evaluation...")
    answer_summary, answer_cases = evaluate_live_workflow(
        benchmark
    )

    observed = {
        "sources": float(len(sources)),
        "corpus_passages": float(len(corpus)),
        "benchmark_questions": float(len(benchmark)),
        "recall_at_3": float(
            retrieval_summary["recall_at_3"]
        ),
        "mean_reciprocal_rank": float(
            retrieval_summary["mean_reciprocal_rank"]
        ),
        "decision_accuracy": required_metric(
            answer_summary,
            "decision_accuracy",
        ),
        "answer_correctness": required_metric(
            answer_summary,
            "answer_correctness",
        ),
        "faithfulness": required_metric(
            answer_summary,
            "faithfulness",
        ),
        "hallucination_rate": required_metric(
            answer_summary,
            "hallucination_rate",
        ),
        "citation_precision": required_metric(
            answer_summary,
            "citation_precision",
        ),
        "citation_recall": required_metric(
            answer_summary,
            "citation_recall",
        ),
        "refusal_accuracy": required_metric(
            answer_summary,
            "refusal_accuracy",
        ),
        "escalation_accuracy": required_metric(
            answer_summary,
            "escalation_accuracy",
        ),
    }

    checks = build_checks(
        observed,
        thresholds,
    )
    failures = []

    print()
    print(
        f"{'Gate':<28}"
        f"{'Observed':>12}"
        f"{'Rule':>8}"
        f"{'Required':>12}"
        f"{'Status':>10}"
    )
    print("-" * 70)

    for name, value, direction, threshold in checks:
        passed = check_passed(
            value,
            direction,
            threshold,
        )
        status = "PASS" if passed else "FAIL"
        rule = ">=" if direction == "minimum" else "<="

        print(
            f"{name:<28}"
            f"{value:>12.4f}"
            f"{rule:>8}"
            f"{threshold:>12.4f}"
            f"{status:>10}"
        )

        if not passed:
            failures.append(
                {
                    "gate": name,
                    "observed": value,
                    "direction": direction,
                    "threshold": threshold,
                }
            )

    failed_question_ids = [
        case["question_id"]
        for case in answer_cases
        if not case["decision_correct"]
        or (
            case["faithfulness"] is not None
            and case["faithfulness"]
            < thresholds["answer_quality"][
                "minimum_faithfulness"
            ]
        )
        or (
            case["citation_precision"] is not None
            and case["citation_precision"]
            < thresholds["answer_quality"][
                "minimum_citation_precision"
            ]
        )
        or (
            case["citation_recall"] is not None
            and case["citation_recall"]
            < thresholds["answer_quality"][
                "minimum_citation_recall"
            ]
        )
    ]

    gate_report = {
        "schema_version": "2.0",
        "status": (
            "passed" if not failures else "failed"
        ),
        "observed": observed,
        "thresholds": thresholds,
        "failures": failures,
        "failed_question_ids": failed_question_ids,
        "retrieval_summary": retrieval_summary,
        "answer_quality_summary": answer_summary,
        "answer_cases": answer_cases,
    }

    output = (
        ROOT
        / "artifacts/runs/ci-quality-gate.json"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(gate_report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Saved gate report to {output}")

    if failed_question_ids:
        print(
            "Cases requiring review:",
            ", ".join(failed_question_ids),
        )

    if failures:
        print()
        print("Quality gate failed:")

        for failure in failures:
            rule = (
                "at least"
                if failure["direction"] == "minimum"
                else "at most"
            )
            print(
                f"- {failure['gate']}: observed "
                f"{failure['observed']}, required {rule} "
                f"{failure['threshold']}"
            )

        raise SystemExit(1)

    print()
    print("All quality gates passed.")


if __name__ == "__main__":
    main()