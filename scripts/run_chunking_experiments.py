"""Compare metadata-preserving chunking configurations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from rag_reliability.data import load_benchmark
from rag_reliability.document_processing import (
    ProcessedChunk,
    build_chunk_manifest,
    chunk_documents,
    load_source_documents,
)
from rag_reliability.retrieval import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT / "artifacts/runs/chunking-comparison.json"
)

CONFIGURATIONS = [
    {
        "name": "heading_aware",
        "strategy": "heading_aware",
        "chunk_size": 0,
        "overlap": 0,
    },
    {
        "name": "fixed_80_20",
        "strategy": "fixed_size",
        "chunk_size": 80,
        "overlap": 20,
    },
    {
        "name": "fixed_120_20",
        "strategy": "fixed_size",
        "chunk_size": 120,
        "overlap": 20,
    },
    {
        "name": "fixed_160_40",
        "strategy": "fixed_size",
        "chunk_size": 160,
        "overlap": 40,
    },
    {
        "name": "recursive_80_20",
        "strategy": "recursive",
        "chunk_size": 80,
        "overlap": 20,
    },
    {
        "name": "recursive_120_20",
        "strategy": "recursive",
        "chunk_size": 120,
        "overlap": 20,
    },
    {
        "name": "recursive_160_40",
        "strategy": "recursive",
        "chunk_size": 160,
        "overlap": 40,
    },
]


def evaluate_configuration(
    chunks: list[ProcessedChunk],
    benchmark: list[Any],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate split chunks using their verified parent passages."""
    retriever = BM25Retriever(chunks)
    eligible = [
        item
        for item in benchmark
        if item.answerable and item.evidence
    ]

    question_results = []
    query_times_ms = []

    for item in eligible:
        relevant_parent_ids = {
            evidence.chunk_id for evidence in item.evidence
        }

        started = perf_counter()
        retrieved = retriever.search(
            item.question,
            top_k=top_k,
        )
        query_times_ms.append(
            (perf_counter() - started) * 1000
        )

        retrieved_chunk_ids = [
            result.chunk.id for result in retrieved
        ]
        retrieved_parent_ids = [
            result.chunk.parent_id for result in retrieved
        ]

        first_relevant_rank = next(
            (
                rank
                for rank, parent_id in enumerate(
                    retrieved_parent_ids,
                    start=1,
                )
                if parent_id in relevant_parent_ids
            ),
            None,
        )

        question_results.append(
            {
                "question_id": item.id,
                "category": item.category,
                "difficulty": item.difficulty,
                "relevant_parent_ids": sorted(
                    relevant_parent_ids
                ),
                "retrieved_chunk_ids": (
                    retrieved_chunk_ids
                ),
                "retrieved_parent_ids": (
                    retrieved_parent_ids
                ),
                "first_relevant_rank": (
                    first_relevant_rank
                ),
            }
        )

    evaluated = len(question_results)
    hits = sum(
        result["first_relevant_rank"] is not None
        for result in question_results
    )
    reciprocal_rank_total = sum(
        1 / result["first_relevant_rank"]
        for result in question_results
        if result["first_relevant_rank"] is not None
    )

    average_chunk_characters = (
        mean(len(chunk.text) for chunk in chunks)
        if chunks
        else 0.0
    )

    return {
        "manifest": build_chunk_manifest(chunks),
        "summary": {
            "evaluated_questions": evaluated,
            f"recall_at_{top_k}": (
                hits / evaluated if evaluated else 0.0
            ),
            "mean_reciprocal_rank": (
                reciprocal_rank_total / evaluated
                if evaluated
                else 0.0
            ),
            "mean_query_time_ms": (
                mean(query_times_ms)
                if query_times_ms
                else 0.0
            ),
            "average_chunk_characters": (
                average_chunk_characters
            ),
        },
        "questions": question_results,
    }


def main() -> None:
    """Run every configured chunking experiment."""
    documents = load_source_documents(
        ROOT / "data/sources.json",
        ROOT / "data/corpus.json",
    )
    benchmark = load_benchmark(
        ROOT / "data/benchmark.json"
    )

    original_passage_count = sum(
        len(document.sections)
        for document in documents
    )
    results = {}

    for configuration in CONFIGURATIONS:
        name = configuration["name"]
        strategy = configuration["strategy"]
        chunk_size = configuration["chunk_size"]
        overlap = configuration["overlap"]

        chunks = chunk_documents(
            documents,
            strategy=strategy,
            chunk_size=chunk_size or 500,
            overlap=overlap,
        )
        report = evaluate_configuration(
            chunks,
            benchmark,
            top_k=3,
        )
        report["configuration"] = configuration
        report["summary"]["index_growth_ratio"] = (
            len(chunks) / original_passage_count
        )
        results[name] = report

    output = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source_registry": "data/sources.json",
        "source_corpus": "data/corpus.json",
        "benchmark": "data/benchmark.json",
        "original_passage_count": original_passage_count,
        "comparison_controls": {
            "retriever": "bm25",
            "top_k": 3,
            "changed_variable": "chunking_configuration",
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Chunking configuration comparison")
    print("-" * 96)
    print(
        f"{'Configuration':22}"
        f"{'Chunks':>9}"
        f"{'Avg chars':>12}"
        f"{'Growth':>10}"
        f"{'Recall@3':>12}"
        f"{'MRR':>10}"
        f"{'Mean ms':>12}"
    )
    print("-" * 96)

    for name, report in results.items():
        manifest = report["manifest"]
        summary = report["summary"]

        print(
            f"{name:22}"
            f"{manifest['chunk_count']:>9}"
            f"{summary['average_chunk_characters']:>12.1f}"
            f"{summary['index_growth_ratio']:>10.2f}"
            f"{summary['recall_at_3']:>12.4f}"
            f"{summary['mean_reciprocal_rank']:>10.4f}"
            f"{summary['mean_query_time_ms']:>12.3f}"
        )

    print()
    print(f"Saved comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()