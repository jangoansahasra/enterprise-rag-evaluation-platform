"""Evaluate answer quality, grounding, citations, and decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag_reliability.answer_evaluation import (
    evaluate_answer,
    summarize_answer_evaluations,
)
from rag_reliability.data import load_benchmark
from rag_reliability.workflow import build_workflow
from run_workflow_demo import build_retriever


ROOT = Path(__file__).resolve().parents[1]


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main() -> None:
    benchmark = load_benchmark(ROOT / "data/benchmark.json")
    workflow = build_workflow(build_retriever())

    cases = []

    for item in benchmark:
        workflow_output = workflow.invoke(
            {"question": item.question}
        )
        case = evaluate_answer(item, workflow_output)
        case["question"] = item.question
        case["verified_answer"] = item.verified_answer
        cases.append(case)

    summary = summarize_answer_evaluations(cases)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_method": {
            "answer_correctness": "normalized lexical token F1",
            "faithfulness": (
                "generated-answer term coverage by retrieved evidence"
            ),
            "citation_metrics": (
                "exact source and section identifier matching"
            ),
            "decision_metrics": (
                "exact match against verified expected behavior"
            ),
            "requires_hosted_model": False,
        },
        "summary": summary,
        "cases": cases,
    }

    output_path = (
        ROOT / "artifacts/runs/answer-quality-evaluation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Answer-quality evaluation")
    print("-" * 46)

    metric_names = [
        "decision_accuracy",
        "answer_correctness",
        "faithfulness",
        "hallucination_rate",
        "citation_precision",
        "citation_recall",
        "refusal_accuracy",
        "escalation_accuracy",
    ]

    for metric_name in metric_names:
        label = metric_name.replace("_", " ").title()
        print(
            f"{label:<26} "
            f"{format_metric(summary[metric_name]):>10}"
        )

    failures = [
        case for case in cases if not case["decision_correct"]
    ]
    print(f"\nDecision failures: {len(failures)}")

    for case in failures:
        print(
            f"- {case['question_id']}: "
            f"expected={case['expected_decision']} "
            f"actual={case['actual_decision']}"
        )

    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()