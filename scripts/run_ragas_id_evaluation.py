"""Run credential-free RAGAS ID-based context evaluation."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from ragas import SingleTurnSample
from ragas.metrics._context_precision import (
    IDBasedContextPrecision,
)
from ragas.metrics._context_recall import (
    IDBasedContextRecall,
)


ROOT = Path(__file__).resolve().parents[1]


async def evaluate_case(
    case: dict,
    precision_metric: IDBasedContextPrecision,
    recall_metric: IDBasedContextRecall,
) -> dict:
    sample = SingleTurnSample(
        retrieved_context_ids=case["retrieved_context_ids"],
        reference_context_ids=case["reference_context_ids"],
    )

    precision = await precision_metric.single_turn_ascore(sample)
    recall = await recall_metric.single_turn_ascore(sample)

    return {
        "question_id": case["question_id"],
        "question": case["question"],
        "retrieved_context_ids": case[
            "retrieved_context_ids"
        ],
        "reference_context_ids": case[
            "reference_context_ids"
        ],
        "id_context_precision": float(precision),
        "id_context_recall": float(recall),
    }


async def async_main() -> None:
    input_path = ROOT / "artifacts/runs/judge-cases.json"

    if not input_path.exists():
        raise SystemExit(
            "Judge cases are missing. Run "
            "`PYTHONPATH=src python "
            "scripts/export_judge_cases.py` first."
        )

    exported = json.loads(input_path.read_text(encoding="utf-8"))
    precision_metric = IDBasedContextPrecision()
    recall_metric = IDBasedContextRecall()

    cases = []

    for case in exported["cases"]:
        cases.append(
            await evaluate_case(
                case,
                precision_metric,
                recall_metric,
            )
        )

    summary = {
        "evaluated_questions": len(cases),
        "mean_id_context_precision": mean(
            case["id_context_precision"] for case in cases
        ),
        "mean_id_context_recall": mean(
            case["id_context_recall"] for case in cases
        ),
    }

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "framework": "ragas",
        "framework_version": version("ragas"),
        "requires_judge_model": False,
        "metric_method": (
            "Exact retrieved and reference context ID comparison"
        ),
        "summary": summary,
        "cases": cases,
    }

    output_path = (
        ROOT / "artifacts/runs/ragas-id-evaluation.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("RAGAS credential-free evaluation")
    print("-" * 42)
    print(
        f"Evaluated questions:       "
        f"{summary['evaluated_questions']}"
    )
    print(
        f"Mean context precision:    "
        f"{summary['mean_id_context_precision']:.4f}"
    )
    print(
        f"Mean context recall:       "
        f"{summary['mean_id_context_recall']:.4f}"
    )
    print(f"\nSaved report to {output_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()