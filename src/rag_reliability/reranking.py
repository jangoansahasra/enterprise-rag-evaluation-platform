"""Cross-encoder reranking for retrieved candidates."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from .retrieval import SearchResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    def __init__(
        self,
        retriever,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        candidate_k: int = 10,
        model=None,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")

        self.retriever = retriever
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.model = model if model is not None else CrossEncoder(model_name)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        candidates = self.retriever.search(
            query,
            top_k=self.candidate_k,
        )

        if not candidates:
            return []

        pairs = [
            (
                query,
                f"{result.chunk.section}\n{result.chunk.text}",
            )
            for result in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
            )
            for result, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda result: (-result.score, result.chunk.id))

        return reranked[:top_k]
