"""Export verified workflow cases for RAGAS and DeepEval."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rag_reliability.data import load_benchmark
from rag_reliability.judge_cases import (
    judge_case_from_workflow,
    to_deepeval_test_case,
    to_ragas_sample,
)
from rag_reliability.workflow import build_workflow
from run_workflow_demo import build_retriever


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-adapters",
        action="store_true",
        help=(
            "Construct RAGAS and DeepEval objects without running "
            "model-based metrics."
        ),
    )
    args = parser.parse_args()

    benchmark = load_benchmark(ROOT / "data/benchmark.json")
    workflow = build_workflow(build_retriever())

    judge_cases = []

    for item in benchmark:
        output = workflow.invoke({"question": item.question})
        case = judge_case_from_workflow(item, output)

        if case is not None:
            judge_cases.append(case)

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source_benchmark": "data/benchmark.json",
        "case_count": len(judge_cases),
        "excluded_behavior": [
            "refuse",
            "escalate",
        ],
        "cases": [case.to_dict() for case in judge_cases],
    }

    output_path = ROOT / "artifacts/runs/judge-cases.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {len(judge_cases)} answerable judge cases.")
    print(f"Saved report to {output_path}")

    if args.validate_adapters:
        ragas_samples = [
            to_ragas_sample(case) for case in judge_cases
        ]
        deepeval_cases = [
            to_deepeval_test_case(case) for case in judge_cases
        ]

        print(
            "RAGAS adapter:",
            len(ragas_samples),
            type(ragas_samples[0]).__name__,
        )
        print(
            "DeepEval adapter:",
            len(deepeval_cases),
            type(deepeval_cases[0]).__name__,
        )


if __name__ == "__main__":
    main()