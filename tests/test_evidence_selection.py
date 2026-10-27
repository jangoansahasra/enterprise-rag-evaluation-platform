from rag_reliability.data import Chunk
from rag_reliability.evidence_selection import (
    HeadingAwareEvidenceSelector,
)
from rag_reliability.retrieval import SearchResult


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        del query
        return self.results[:top_k]


def result(
    chunk_id: str,
    section: str,
    text: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            source_id="source",
            section=section,
            text=text,
        ),
        score=score,
    )


def test_heading_and_passage_terms_break_close_rank_tie() -> None:
    selector = HeadingAwareEvidenceSelector(
        StubRetriever(
            [
                result(
                    "general",
                    "Suspicious activity",
                    "Ask the security team for help.",
                    0.99,
                ),
                result(
                    "gmail",
                    "Submission of phishing email",
                    "Select the reporting icon in Gmail.",
                    0.98,
                ),
            ]
        )
    )

    selected = selector.search(
        "How do I report a suspicious email in Gmail?",
        top_k=2,
    )

    assert selected[0].chunk.id == "gmail"


def test_selector_can_promote_specific_lower_ranked_evidence() -> None:
    selector = HeadingAwareEvidenceSelector(
        StubRetriever(
            [
                result(
                    "primary",
                    "AI data rules",
                    "Customer credentials are RED data.",
                    0.9,
                ),
                result(
                    "unrelated",
                    "Manager approval",
                    "Most requests require approval.",
                    0.8,
                ),
                result(
                    "credentials",
                    "Credentials and access tokens",
                    "Credentials inherit the protected data level.",
                    0.7,
                ),
            ]
        )
    )

    selected = selector.search(
        "Can customer credentials be used in an AI tool?",
        top_k=3,
    )

    assert selected[0].chunk.id == "primary"
    assert (
        selected.index(
            next(
                item
                for item in selected
                if item.chunk.id == "credentials"
            )
        )
        < selected.index(
            next(
                item
                for item in selected
                if item.chunk.id == "unrelated"
            )
        )
    )
    
def test_selector_preserves_confident_original_top_result() -> None:
    selector = HeadingAwareEvidenceSelector(
        StubRetriever(
            [
                result(
                    "confident-primary",
                    "Device security",
                    "Report a lost laptop immediately.",
                    6.0,
                ),
                result(
                    "lexical-distractor",
                    "Lost laptop company credentials",
                    (
                        "Lost laptop company credentials security "
                        "reporting guidance."
                    ),
                    2.0,
                ),
            ]
        ),
        maximum_score_drop=0.25,
    )

    selected = selector.search(
        "How do I report a lost laptop with company credentials?",
        top_k=2,
    )

    assert selected[0].chunk.id == "confident-primary"