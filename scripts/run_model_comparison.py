"""Compare local models on verified enterprise RAG cases."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from rag_reliability.answer_evaluation import (
    citation_scores,
    evidence_support,
    token_f1,
)
from rag_reliability.model_clients import OllamaClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/runs/judge-cases.json"
DEFAULT_OUTPUT = ROOT / "artifacts/runs/model-comparison.json"

DEFAULT_MODELS = [
    "llama3.2:3b-instruct-q4_K_M",
    "mistral:7b",
    "phi4-mini:latest",
]


def available_mean(
    values: Iterable[float | int | None],
) -> float | None:
    """Return the mean of available numeric values."""
    available = [
        float(value)
        for value in values
        if value is not None
    ]
    return mean(available) if available else None


def evaluate_result(
    case: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Calculate deterministic quality metrics for one model result."""
    answer_correctness = None
    faithfulness = None
    citation_precision = None
    citation_recall = None

    if result.structured_output_valid:
        answer_correctness = token_f1(
            result.answer,
            str(case["expected_output"]),
        )
        faithfulness = evidence_support(
            result.answer,
            [
                str(context)
                for context in case["retrieval_context"]
            ],
        )
        citation_precision, citation_recall = citation_scores(
            list(result.citations),
            [
                str(citation)
                for citation in case["reference_context_ids"]
            ],
        )

    return {
        "question_id": case["question_id"],
        "question": case["question"],
        "expected_output": case["expected_output"],
        "expected_citations": case["reference_context_ids"],
        "answer": result.answer,
        "citations": list(result.citations),
        "structured_output_valid": (
            result.structured_output_valid
        ),
        "answer_correctness": answer_correctness,
        "faithfulness": faithfulness,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "error": result.error,
        "raw_output": result.raw_output,
    }


def summarize_cases(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate quality, reliability, latency, and usage metrics."""
    evaluated = len(cases)
    valid_count = sum(
        bool(case["structured_output_valid"])
        for case in cases
    )
    failure_count = sum(
        case["error"] is not None for case in cases
    )

    return {
        "evaluated_questions": evaluated,
        "structured_output_reliability": (
            valid_count / evaluated if evaluated else 0.0
        ),
        "mean_answer_correctness": available_mean(
            case["answer_correctness"] for case in cases
        ),
        "mean_faithfulness": available_mean(
            case["faithfulness"] for case in cases
        ),
        "mean_citation_precision": available_mean(
            case["citation_precision"] for case in cases
        ),
        "mean_citation_recall": available_mean(
            case["citation_recall"] for case in cases
        ),
        "mean_latency_ms": available_mean(
            case["latency_ms"] for case in cases
        ),
        "total_input_tokens": sum(
            int(case["input_tokens"] or 0)
            for case in cases
        ),
        "total_output_tokens": sum(
            int(case["output_tokens"] or 0)
            for case in cases
        ),
        "estimated_cost_usd": sum(
            float(case["estimated_cost_usd"])
            for case in cases
        ),
        "failure_count": failure_count,
    }


def build_report(
    source_path: Path,
    requested_models: list[str],
    results: dict[str, list[dict[str, Any]]],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Build a serializable comparison report."""
    model_reports = []

    for model in requested_models:
        cases = results.get(model, [])
        model_reports.append(
            {
                "provider": "ollama",
                "model": model,
                "deployment": "local",
                "cases": cases,
                "summary": summarize_cases(cases),
            }
        )

    return {
        "schema_version": "1.0",
        "updated_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "source_cases": str(source_path.relative_to(ROOT)),
        "measurement_notes": [
            "Models run locally through Ollama.",
            "Estimated API cost is zero for local inference.",
            "Latency excludes the unrecorded warm-up request.",
            "Quality metrics are deterministic proxies.",
        ],
        "models": model_reports,
    }


def save_report(
    output_path: Path,
    source_path: Path,
    requested_models: list[str],
    results: dict[str, list[dict[str, Any]]],
    *,
    complete: bool,
) -> None:
    """Write current progress after each evaluated case."""
    report = build_report(
        source_path,
        requested_models,
        results,
        complete=complete,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact terminal comparison."""
    print()
    print("Local model comparison")
    print("-" * 104)
    print(
        f"{'Model':34}"
        f"{'Valid':>9}"
        f"{'Correct':>10}"
        f"{'Faithful':>10}"
        f"{'Cite P':>10}"
        f"{'Cite R':>10}"
        f"{'Mean ms':>12}"
        f"{'Failures':>9}"
    )
    print("-" * 104)

    for model_report in report["models"]:
        summary = model_report["summary"]

        def display(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        print(
            f"{model_report['model'][:33]:34}"
            f"{summary['structured_output_reliability']:>9.4f}"
            f"{display(summary['mean_answer_correctness']):>10}"
            f"{display(summary['mean_faithfulness']):>10}"
            f"{display(summary['mean_citation_precision']):>10}"
            f"{display(summary['mean_citation_recall']):>10}"
            f"{display(summary['mean_latency_ms']):>12}"
            f"{summary['failure_count']:>9}"
        )


def main() -> None:
    """Run local models over the shared verified judge cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Ollama model name. Repeat to compare multiple models. "
            "Defaults to the three configured local models."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N cases for a smoke test.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the exported shared judge cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the comparison report.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the unrecorded model warm-up request.",
    )
    args = parser.parse_args()

    source_path = args.input.resolve()
    output_path = args.output.resolve()
    requested_models = args.models or DEFAULT_MODELS

    if not source_path.exists():
        raise SystemExit(
            "Judge cases are missing. Run "
            "`PYTHONPATH=src python scripts/export_judge_cases.py` "
            "first."
        )

    source_report = json.loads(
        source_path.read_text(encoding="utf-8")
    )
    cases = list(source_report["cases"])

    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1.")
        cases = cases[: args.limit]

    results: dict[str, list[dict[str, Any]]] = {
        model: [] for model in requested_models
    }

    for model in requested_models:
        print(f"\nLoading model: {model}")
        client = OllamaClient(model)

        if not args.skip_warmup and cases:
            warmup_case = cases[0]
            warmup_contexts = [
                {
                    "id": str(context_id),
                    "text": str(context_text),
                }
                for context_id, context_text in zip(
                    warmup_case["retrieved_context_ids"],
                    warmup_case["retrieval_context"],
                    strict=True,
                )
            ]
            print("Running unrecorded warm-up request...")
            client.generate(
                str(warmup_case["question"]),
                warmup_contexts,
            )

        for index, case in enumerate(cases, start=1):
            contexts = [
                {
                    "id": str(context_id),
                    "text": str(context_text),
                }
                for context_id, context_text in zip(
                    case["retrieved_context_ids"],
                    case["retrieval_context"],
                    strict=True,
                )
            ]

            print(
                f"[{model}] {index}/{len(cases)} "
                f"{case['question_id']}"
            )
            model_result = client.generate(
                str(case["question"]),
                contexts,
            )
            evaluated = evaluate_result(case, model_result)
            results[model].append(evaluated)

            save_report(
                output_path,
                source_path,
                requested_models,
                results,
                complete=False,
            )

    save_report(
        output_path,
        source_path,
        requested_models,
        results,
        complete=True,
    )

    final_report = build_report(
        source_path,
        requested_models,
        results,
        complete=True,
    )
    print_summary(final_report)
    print()
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()