from rag_reliability.data import Chunk
from rag_reliability.retrieval import SearchResult
from rag_reliability.workflow import build_workflow


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls = 0

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        del query
        self.calls += 1
        return self.results[:top_k]


def result(
    chunk_id: str,
    section: str,
    text: str,
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            source_id="gitlab-policy",
            section=section,
            text=text,
        ),
        score=score,
    )


def test_supported_question_returns_cited_evidence() -> None:
    retriever = StubRetriever(
        [
            result(
                "password-policy",
                "Passwords at GitLab",
                "GitLab utilizes 1Password for password management.",
            )
        ]
    )
    workflow = build_workflow(retriever)

    output = workflow.invoke(
        {"question": "Which password manager does GitLab use?"}
    )

    assert output["decision"] == "answer"
    assert output["citations"] == ["password-policy"]
    assert "1Password" in output["answer"]
    assert retriever.calls == 1


def test_out_of_domain_question_is_refused_without_retrieval() -> None:
    retriever = StubRetriever([])
    workflow = build_workflow(retriever)

    output = workflow.invoke(
        {"question": "What is the weather forecast tomorrow?"}
    )

    assert output["decision"] == "refuse"
    assert output["citations"] == []
    assert retriever.calls == 0


def test_authorization_request_is_escalated_without_retrieval() -> None:
    retriever = StubRetriever([])
    workflow = build_workflow(retriever)

    output = workflow.invoke(
        {"question": "Grant me permanent production database access."}
    )

    assert output["decision"] == "escalate"
    assert "authorization" in output["reason"].lower()
    assert retriever.calls == 0


def test_unsupported_claim_is_refused() -> None:
    retriever = StubRetriever(
        [
            result(
                "unrelated",
                "Password guidance",
                "Use a password manager.",
            )
        ]
    )
    workflow = build_workflow(retriever)

    output = workflow.invoke(
        {"question": "What is the guaranteed laptop replacement SLA?"}
    )

    assert output["decision"] == "refuse"
    assert output["evidence_quality"] == "insufficient"
    assert output["citations"] == []


def test_term_normalization_handles_common_word_forms() -> None:
    from rag_reliability.workflow import meaningful_terms

    assert "coordinat" in meaningful_terms("coordinates coordination")
    assert "incident" in meaningful_terms("incident incidents")
    assert "credential" in meaningful_terms("credential credentials")
