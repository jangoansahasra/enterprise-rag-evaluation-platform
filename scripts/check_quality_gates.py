"""Apply deterministic CI quality gates to the current project."""

from __future__ import annotations

import json
from pathlib import Path

from rag_reliability.data import load_benchmark, load_corpus
from rag_reliability.evaluation import evaluate_retriever
from rag_reliability.retrieval import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    thresholds = load_json(ROOT / "config/quality-gates.json")
    sources = load_json(ROOT / "data/sources.json")
    benchmark = load_benchmark(ROOT / "data/benchmark.json")
    corpus = load_corpus(ROOT / "data/corpus.json")

    retrieval_report = evaluate_retriever(
        BM25Retriever(corpus),
        benchmark,
        top_k=3,
    )
    summary = retrieval_report["summary"]

    observed = {
        "sources": len(sources),
        "corpus_passages": len(corpus),
        "benchmark_questions": len(benchmark),
        "recall_at_3": summary["recall_at_3"],
        "mean_reciprocal_rank": summary["mean_reciprocal_rank"],
    }

    failures: list[str] = []

    dataset_gates = thresholds["dataset"]
    retrieval_gates = thresholds["retrieval"]

    checks = [
        (
            "sources",
            observed["sources"],
            dataset_gates["minimum_sources"],
        ),
        (
            "corpus_passages",
            observed["corpus_passages"],
            dataset_gates["minimum_corpus_passages"],
        ),
        (
            "benchmark_questions",
            observed["benchmark_questions"],
            dataset_gates["minimum_benchmark_questions"],
        ),
        (
            "recall_at_3",
            observed["recall_at_3"],
            retrieval_gates["minimum_recall_at_3"],
        ),
        (
            "mean_reciprocal_rank",
            observed["mean_reciprocal_rank"],
            retrieval_gates["minimum_mean_reciprocal_rank"],
        ),
    ]

    print(f"{'Gate':<28} {'Observed':>12} {'Required':>12} {'Status':>10}")
    print("-" * 66)

    for name, value, minimum in checks:
        passed = value >= minimum
        status = "PASS" if passed else "FAIL"

        print(
            f"{name:<28} "
            f"{value:>12.4f} "
            f"{minimum:>12.4f} "
            f"{status:>10}"
        )

        if not passed:
            failures.append(
                f"{name}: observed {value}, required at least {minimum}"
            )

    gate_report = {
        "schema_version": "1.0",
        "status": "passed" if not failures else "failed",
        "observed": observed,
        "thresholds": thresholds,
        "failures": failures,
    }

    output = ROOT / "artifacts/runs/ci-quality-gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(gate_report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved gate report to {output}")

    if failures:
        print("\nQuality gate failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nAll quality gates passed.")


if __name__ == "__main__":
    main()