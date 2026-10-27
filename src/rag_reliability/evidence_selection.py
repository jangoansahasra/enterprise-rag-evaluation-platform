"""Heading-aware evidence selection over retrieved candidates."""

from __future__ import annotations

from rag_reliability.retrieval import SearchResult
from rag_reliability.workflow import meaningful_terms


def overlap_ratio(query_terms: set[str], text: str) -> float:
    """Return the proportion of query terms present in text."""
    if not query_terms:
        return 0.0

    return len(query_terms & meaningful_terms(text)) / len(query_terms)

def lexical_coverage_score(
    query: str,
    result: SearchResult,
) -> float:
    """Score heading and passage coverage without rank influence."""
    query_terms = meaningful_terms(query)
    heading_overlap = overlap_ratio(
        query_terms,
        result.chunk.section,
    )
    passage_overlap = overlap_ratio(
        query_terms,
        result.chunk.text,
    )

    return 0.45 * heading_overlap + 0.40 * passage_overlap


def selection_score(
    query: str,
    result: SearchResult,
    rank: int,
) -> float:
    """Combine original rank with lexical evidence coverage."""
    if rank < 1:
        raise ValueError("rank must be positive")

    reciprocal_rank = 1.0 / rank
    return (
        0.15 * reciprocal_rank
        + lexical_coverage_score(query, result)
    )


class HeadingAwareEvidenceSelector:
    """Rerank candidates using headings and exact query coverage."""

    def __init__(
        self,
        retriever,
        *,
        candidate_k: int = 5,
        promotion_threshold: float = 0.05,
        maximum_score_drop: float = 0.25,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")

        self.retriever = retriever
        self.candidate_k = candidate_k
        self.promotion_threshold = promotion_threshold
        self.maximum_score_drop = maximum_score_drop

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        candidates = self.retriever.search(
            query,
            top_k=max(top_k, self.candidate_k),
        )

        if not candidates:
            return []

        scored = [
            (
                result,
                selection_score(query, result, rank),
                lexical_coverage_score(query, result),
            )
            for rank, result in enumerate(candidates, start=1)
        ]

        original_top = scored[0]
        lexical_best = max(
            scored,
            key=lambda item: (
                item[2],
                item[1],
                item[0].chunk.id,
            ),
        )

        if (
            lexical_best[2]
            >= original_top[2] + self.promotion_threshold
            and lexical_best[0].score
            >= original_top[0].score - self.maximum_score_drop
        ):
            selected_first = lexical_best
        else:
            selected_first = original_top

        remaining = [
            item for item in scored if item is not selected_first
        ]
        remaining.sort(
            key=lambda item: (-item[1], item[0].chunk.id)
        )
        ordered = [selected_first, *remaining]

        return [
            SearchResult(
                chunk=result.chunk,
                score=score,
            )
            for result, score, _ in ordered[:top_k]
        ]