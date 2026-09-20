import pytest

from rag_reliability.data import Chunk
from rag_reliability.reranking import CrossEncoderReranker
from rag_reliability.retrieval import SearchResult


class StubRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        del query
        return [
            SearchResult(chunk=chunk, score=float(len(self.chunks) - rank))
            for rank, chunk in enumerate(self.chunks[:top_k])
        ]


class StubCrossEncoder:
    def predict(self, pairs, *, show_progress_bar: bool = False):
        assert show_progress_bar is False
        assert len(pairs) == 2
        return [0.1, 0.9]


def test_cross_encoder_reorders_candidates() -> None:
    first = Chunk("first", "source", "First section", "less relevant")
    second = Chunk("second", "source", "Second section", "more relevant")

    reranker = CrossEncoderReranker(
        StubRetriever([first, second]),
        candidate_k=2,
        model=StubCrossEncoder(),
    )

    results = reranker.search("test query", top_k=2)

    assert [result.chunk.id for result in results] == ["second", "first"]
    assert results[0].score == pytest.approx(0.9)


def test_reranker_rejects_invalid_candidate_count() -> None:
    with pytest.raises(ValueError, match="candidate_k"):
        CrossEncoderReranker(
            StubRetriever([]),
            candidate_k=0,
            model=StubCrossEncoder(),
        )
