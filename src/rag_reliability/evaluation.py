"""Deterministic retrieval evaluation against source-verified evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .data import BenchmarkItem
from .retrieval import BM25Retriever


@dataclass(frozen=True)
class QueryResult:
    question_id: str
    category: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    first_relevant_rank: int | None


def evaluate_retriever(
    retriever: BM25Retriever, items: list[BenchmarkItem], *, top_k: int = 3
) -> dict:
    eligible = [item for item in items if item.answerable and item.evidence]
    query_results: list[QueryResult] = []
    for item in eligible:
        relevant = {evidence.chunk_id for evidence in item.evidence}
        retrieved = retriever.search(item.question, top_k=top_k)
        retrieved_ids = [result.chunk.id for result in retrieved]
        first_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in relevant),
            None,
        )
        query_results.append(
            QueryResult(
                question_id=item.id,
                category=item.category,
                relevant_chunk_ids=sorted(relevant),
                retrieved_chunk_ids=retrieved_ids,
                first_relevant_rank=first_rank,
            )
        )

    total = len(query_results)
    hits = sum(result.first_relevant_rank is not None for result in query_results)
    reciprocal_rank = sum(
        1 / result.first_relevant_rank
        for result in query_results
        if result.first_relevant_rank is not None
    )
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {"retriever": "bm25", "top_k": top_k},
        "summary": {
            "evaluated_questions": total,
            f"recall_at_{top_k}": hits / total if total else 0.0,
            "mean_reciprocal_rank": reciprocal_rank / total if total else 0.0,
        },
        "questions": [asdict(result) for result in query_results],
    }
