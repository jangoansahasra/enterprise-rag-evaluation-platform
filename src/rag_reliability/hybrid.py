"""Hybrid retrieval using reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from .data import Chunk
from .retrieval import SearchResult


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return ranked results for a query."""


class HybridRetriever:
    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        *,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        sparse_results = self.sparse_retriever.search(
            query,
            top_k=self.candidate_k,
        )
        dense_results = self.dense_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        fused_scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}

        for results in (sparse_results, dense_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.id
                chunks[chunk_id] = result.chunk
                fused_scores[chunk_id] += 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )

        return [
            SearchResult(
                chunk=chunks[chunk_id],
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_ids[:top_k]
        ]
