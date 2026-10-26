from rag_reliability.answer_evaluation import (
    citation_scores,
    evaluate_answer,
    evidence_support,
    summarize_answer_evaluations,
    token_f1,
)
from rag_reliability.data import (
    BenchmarkItem,
    Chunk,
    Evidence,
)
from rag_reliability.retrieval import SearchResult


def answerable_item() -> BenchmarkItem:
    return BenchmarkItem(
        id="q-test-answer",
        question="Which password manager does GitLab use?",
        category="password",
        difficulty="direct",
        answerable=True,
        expected_behavior="answer",
        verified_answer=(
            "GitLab uses 1Password for password management."
        ),
        evidence=(
            Evidence(
                source_id="gitlab-password-guidelines",
                section="Passwords at GitLab",
                supporting_passage=(
                    "GitLab utilizes 1Password for password management."
                ),
            ),
        ),
    )


def refusal_item() -> BenchmarkItem:
    return BenchmarkItem(
        id="q-test-refuse",
        question="What is the guaranteed laptop replacement SLA?",
        category="service_level",
        difficulty="unanswerable",
        answerable=False,
        expected_behavior="refuse",
        verified_answer=None,
        evidence=(),
    )


def retrieved_result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=(
                "gitlab-password-guidelines"
                "::Passwords at GitLab"
            ),
            source_id="gitlab-password-guidelines",
            section="Passwords at GitLab",
            text=(
                "GitLab utilizes 1Password for password management."
            ),
        ),
        score=1.0,
    )


def test_token_f1_rewards_matching_answer() -> None:
    score = token_f1(
        "GitLab uses 1Password for password management.",
        "GitLab uses 1Password for password management.",
    )

    assert score == 1.0


def test_evidence_support_detects_unsupported_terms() -> None:
    fully_supported = evidence_support(
        "Use 1Password for password management.",
        ["Use 1Password for password management."],
    )
    partly_supported = evidence_support(
        "Use 1Password and receive guaranteed phone support.",
        ["Use 1Password for password management."],
    )

    assert fully_supported == 1.0
    assert partly_supported < fully_supported


def test_citation_scores_penalize_wrong_source() -> None:
    precision, recall = citation_scores(
        ["wrong-source::Wrong section"],
        ["expected-source::Expected section"],
    )

    assert precision == 0.0
    assert recall == 0.0


def test_supported_answer_receives_grounding_scores() -> None:
    item = answerable_item()
    output = {
        "decision": "answer",
        "answer": (
            "GitLab utilizes 1Password for password management."
        ),
        "citations": [item.evidence[0].chunk_id],
        "retrieved": [retrieved_result()],
        "reason": "Evidence passed the support threshold.",
    }

    result = evaluate_answer(item, output)

    assert result["decision_correct"] is True
    assert result["faithfulness"] == 1.0
    assert result["hallucination_rate"] == 0.0
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 1.0
    assert result["answer_correctness"] > 0.5


def test_correct_refusal_has_no_generation_metrics() -> None:
    output = {
        "decision": "refuse",
        "answer": "The claim is not supported by the knowledge base.",
        "citations": [],
        "retrieved": [],
        "reason": "Retrieved evidence did not support the claim.",
    }

    result = evaluate_answer(refusal_item(), output)

    assert result["decision_correct"] is True
    assert result["refusal_correct"] is True
    assert result["answer_correctness"] is None
    assert result["faithfulness"] is None
    assert result["hallucination_rate"] is None


def test_summary_uses_only_applicable_metrics() -> None:
    cases = [
        {
            "decision_correct": True,
            "answer_correctness": 0.8,
            "faithfulness": 1.0,
            "hallucination_rate": 0.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "refusal_correct": None,
            "escalation_correct": None,
        },
        {
            "decision_correct": True,
            "answer_correctness": None,
            "faithfulness": None,
            "hallucination_rate": None,
            "citation_precision": None,
            "citation_recall": None,
            "refusal_correct": True,
            "escalation_correct": None,
        },
    ]

    summary = summarize_answer_evaluations(cases)

    assert summary["evaluated_questions"] == 2
    assert summary["decision_accuracy"] == 1.0
    assert summary["answer_correctness"] == 0.8
    assert summary["faithfulness"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["escalation_accuracy"] is None