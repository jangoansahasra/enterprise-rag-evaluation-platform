from rag_reliability.data import BenchmarkItem, Evidence
from rag_reliability.judge_cases import (
    judge_case_from_workflow,
)


def benchmark_item(
    *,
    expected_behavior: str = "answer",
    verified_answer: str | None = "Use the approved process.",
) -> BenchmarkItem:
    return BenchmarkItem(
        id="q-judge",
        question="What process should I use?",
        category="support",
        difficulty="direct",
        answerable=expected_behavior == "answer",
        expected_behavior=expected_behavior,
        verified_answer=verified_answer,
        evidence=(
            Evidence(
                source_id="policy",
                section="Approved process",
                supporting_passage="Use the approved process.",
            ),
        ),
    )


def workflow_output(
    *,
    decision: str = "answer",
) -> dict:
    return {
        "decision": decision,
        "answer": "Use the approved process.",
        "citations": ["policy::Approved process"],
        "retrieved_evidence": [
            {
                "chunk_id": "policy::Approved process",
                "source_id": "policy",
                "section": "Approved process",
                "text": "Use the approved process.",
                "score": 1.0,
            }
        ],
        "reason": "Evidence passed.",
    }


def test_answerable_workflow_result_creates_judge_case() -> None:
    case = judge_case_from_workflow(
        benchmark_item(),
        workflow_output(),
    )

    assert case is not None
    assert case.question_id == "q-judge"
    assert case.actual_output == "Use the approved process."
    assert case.expected_output == "Use the approved process."
    assert case.retrieval_context == (
        "Use the approved process.",
    )
    assert case.reference_context == (
        "Use the approved process.",
    )
    assert case.retrieved_context_ids == (
        "policy::Approved process",
    )
    assert case.reference_context_ids == (
        "policy::Approved process",
    )
    assert case.citations == ("policy::Approved process",)


def test_refusal_is_excluded_from_generation_judges() -> None:
    case = judge_case_from_workflow(
        benchmark_item(
            expected_behavior="refuse",
            verified_answer=None,
        ),
        workflow_output(decision="refuse"),
    )

    assert case is None


def test_incorrect_workflow_decision_is_excluded() -> None:
    case = judge_case_from_workflow(
        benchmark_item(),
        workflow_output(decision="refuse"),
    )

    assert case is None