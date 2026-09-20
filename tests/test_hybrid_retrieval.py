import pytest

from rag_reliability.data import Chunk
from rag_reliability.dense import DenseRetriever
from rag_reliability.hybrid import HybridRetriever
from rag_reliability.retrieval import SearchResult


class StubRetriever:
    def __init__(self, ranked_chunks: list[Chunk]) -> None:
        self.ranked_chunks = ranked_chunks

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        del query
        return [
            SearchResult(chunk=chunk, score=float(len(self.ranked_chunks) - rank))
            for rank, chunk in enumerate(self.ranked_chunks[:top_k])
        ]


def test_rrf_promotes_a_chunk_found_by_both_retrievers() -> None:
    chunk_a = Chunk("a", "source", "A", "alpha")
    chunk_b = Chunk("b", "source", "B", "beta")
    chunk_c = Chunk("c", "source", "C", "gamma")

    hybrid = HybridRetriever(
        StubRetriever([chunk_a, chunk_b]),
        StubRetriever([chunk_b, chunk_c]),
        candidate_k=3,
    )

    results = hybrid.search("query", top_k=3)

    assert results[0].chunk.id == "b"
    assert {result.chunk.id for result in results} == {"a", "b", "c"}


def test_hybrid_rejects_invalid_configuration() -> None:
    empty = StubRetriever([])

    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetriever(empty, empty, candidate_k=0)


def test_dense_retriever_requires_a_corpus() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        DenseRetriever([])
