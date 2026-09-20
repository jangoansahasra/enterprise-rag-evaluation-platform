"""Evaluate workflow answer, refusal, and escalation decisions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rag_reliability.data import load_benchmark
from rag_reliability.workflow import build_workflow
from run_workflow_demo import build_retriever


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    benchmark = load_benchmark(ROOT / "data/benchmark.json")
    workflow = build_workflow(build_retriever())

    cases = []

    for item in benchmark:
        output = workflow.invoke({"question": item.question})
        actual = output["decision"]
        expected = item.expected_behavior

        cases.append(
            {
                "question_id": item.id,
                "question": item.question,
                "expected_decision": expected,
                "actual_decision": actual,
                "correct": actual == expected,
                "classification": output["classification"],
                "evidence_quality": output.get("evidence_quality"),
                "evidence_overlap": output.get("evidence_overlap"),
                "required_claim_terms": output.get(
                    "required_claim_terms",
                    [],
                ),
                "unsupported_claim_terms": output.get(
                    "unsupported_claim_terms",
                    [],
                ),
                "citations": output["citations"],
                "reason": output["reason"],
            }
        )

    correct = sum(case["correct"] for case in cases)
    total = len(cases)

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "evaluated_questions": total,
            "correct_decisions": correct,
            "decision_accuracy": correct / total if total else 0.0,
        },
        "cases": cases,
    }

    output_path = ROOT / "artifacts/runs/workflow-decisions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"Decision accuracy: {correct}/{total} "
        f"= {report['summary']['decision_accuracy']:.1%}"
    )

    failures = [case for case in cases if not case["correct"]]
    print("\nFailures:")

    for case in failures:
        print(
            case["question_id"],
            f"expected={case['expected_decision']}",
            f"actual={case['actual_decision']}",
            f"overlap={case['evidence_overlap']}",
            f"unsupported={case['unsupported_claim_terms']}",
        )
        print(" ", case["question"])

    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
