"""Compare sparse, dense, and hybrid retrieval configurations."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from rag_reliability.tracking import log_retrieval_experiment
from rag_reliability.data import load_benchmark, load_corpus
from rag_reliability.dense import DEFAULT_EMBEDDING_MODEL, DenseRetriever
from rag_reliability.evaluation import evaluate_retriever
from rag_reliability.hybrid import HybridRetriever
from rag_reliability.reranking import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from rag_reliability.retrieval import BM25Retriever
from rag_reliability.observability import (
    configure_observability,
    log_event,
    observed_operation,
)


ROOT = Path(__file__).resolve().parents[1]
TOP_K = 3


def timed_evaluation(name: str, retriever, benchmark: list) -> dict:
    start = perf_counter()
    report = evaluate_retriever(retriever, benchmark, top_k=TOP_K)
    elapsed_ms = (perf_counter() - start) * 1000

    report["configuration"] = {"retriever": name, "top_k": TOP_K}
    report["timing"] = {
        "total_query_time_ms": round(elapsed_ms, 3),
        "mean_query_time_ms": round(
            elapsed_ms / report["summary"]["evaluated_questions"],
            3,
        ),
    }
    return report


def main() -> None:
    benchmark = load_benchmark(ROOT / "data/benchmark.json")
    corpus = load_corpus(ROOT / "data/corpus.json")
    logger, tracer = configure_observability(
    ROOT / "artifacts/observability"
    )

    build_start = perf_counter()
    bm25 = BM25Retriever(corpus)
    bm25_build_ms = (perf_counter() - build_start) * 1000

    build_start = perf_counter()
    dense = DenseRetriever(corpus)
    dense_build_ms = (perf_counter() - build_start) * 1000

    hybrid = HybridRetriever(
        bm25,
        dense,
        candidate_k=10,
        rrf_k=60,
    )

    build_start = perf_counter()
    reranked = CrossEncoderReranker(
        hybrid,
        candidate_k=10,
    )
    reranker_build_ms = (perf_counter() - build_start) * 1000

    retrievers = {
        "bm25": bm25,
        "dense": dense,
        "hybrid_rrf": hybrid,
        "hybrid_reranked": reranked,
    }
    reports = {}

    for name, retriever in retrievers.items():
        with observed_operation(
            tracer,
            logger,
            "evaluate_retriever",
            attributes={
                "retriever.name": name,
                "evaluation.top_k": TOP_K,
                "evaluation.question_count": len(benchmark),
            },
        ) as request_id:
            report = timed_evaluation(
                name,
                retriever,
                benchmark,
            )
            reports[name] = report

            log_event(
                logger,
                "retrieval_evaluation_completed",
                request_id=request_id,
                retriever=name,
                evaluated_questions=report["summary"][
                    "evaluated_questions"
                ],
                recall_at_3=report["summary"]["recall_at_3"],
                mean_reciprocal_rank=report["summary"][
                    "mean_reciprocal_rank"
                ],
                mean_query_time_ms=report["timing"][
                    "mean_query_time_ms"
                ],
            )

    reports["bm25"]["configuration"]["index_build_ms"] = round(
        bm25_build_ms,
        3,
    )
    reports["dense"]["configuration"].update(
        {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "index": "faiss.IndexFlatIP",
            "index_build_ms": round(dense_build_ms, 3),
            "metadata_enriched": True,
        }
    )
    reports["hybrid_rrf"]["configuration"].update(
        {
            "candidate_k": 10,
            "rrf_k": 60,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
        }
    )
    reports["hybrid_reranked"]["configuration"].update(
        {
            "candidate_k": 10,
            "reranker_model": DEFAULT_RERANKER_MODEL,
            "model_build_ms": round(reranker_build_ms, 3),
        }
    )

    output = ROOT / "artifacts/runs/retrieval-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")
    run_ids = log_retrieval_experiment(
    reports,
    ROOT / "artifacts/mlflow",
    )

    print(f"{'Retriever':<14} {'Recall@3':>10} {'MRR':>10} {'Mean ms':>12}")
    print("-" * 50)

    for name, report in reports.items():
        summary = report["summary"]
        timing = report["timing"]
        print(
            f"{name:<14} "
            f"{summary['recall_at_3']:>10.4f} "
            f"{summary['mean_reciprocal_rank']:>10.4f} "
            f"{timing['mean_query_time_ms']:>12.3f}"
        )

    print(f"\nSaved comparison to {output}")
    print("\nMLflow runs:")
    for name, run_id in run_ids.items():
        print(f"- {name}: {run_id}")


if __name__ == "__main__":
    main()