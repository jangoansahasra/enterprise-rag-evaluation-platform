"""Shared evaluation cases for optional judge-based frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rag_reliability.data import BenchmarkItem


@dataclass(frozen=True)
class JudgeCase:
    """One answerable case prepared for external evaluators."""

    question_id: str
    question: str
    actual_output: str
    expected_output: str
    retrieval_context: tuple[str, ...]
    reference_context: tuple[str, ...]
    retrieved_context_ids: tuple[str, ...]
    reference_context_ids: tuple[str, ...]
    citations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def judge_case_from_workflow(
    benchmark_item: BenchmarkItem,
    workflow_output: dict[str, Any],
) -> JudgeCase | None:
    """Create a judge case when a verified answer is available."""
    if (
        benchmark_item.expected_behavior != "answer"
        or not benchmark_item.verified_answer
        or workflow_output["decision"] != "answer"
    ):
        return None

    retrieved = workflow_output.get("retrieved_evidence", [])
    retrieval_context = tuple(
        evidence["text"] for evidence in retrieved
    )
    retrieved_context_ids = tuple(
        evidence["chunk_id"] for evidence in retrieved
    )
    reference_context = tuple(
        evidence.supporting_passage
        for evidence in benchmark_item.evidence
    )
    reference_context_ids = tuple(
        evidence.chunk_id for evidence in benchmark_item.evidence
    )

    return JudgeCase(
        question_id=benchmark_item.id,
        question=benchmark_item.question,
        actual_output=workflow_output["answer"],
        expected_output=benchmark_item.verified_answer,
        retrieval_context=retrieval_context,
        reference_context=reference_context,
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids,
        citations=tuple(workflow_output["citations"]),
    )


def to_ragas_sample(case: JudgeCase):
    """Convert a shared case to a RAGAS SingleTurnSample."""
    try:
        from ragas import SingleTurnSample
    except ImportError as error:
        raise RuntimeError(
            "Install judge dependencies with "
            '`python -m pip install -e ".[judge]"`.'
        ) from error

    return SingleTurnSample(
        user_input=case.question,
        response=case.actual_output,
        reference=case.expected_output,
        retrieved_contexts=list(case.retrieval_context),
        reference_contexts=list(case.reference_context),
        retrieved_context_ids=list(
            case.retrieved_context_ids
        ),
        reference_context_ids=list(
            case.reference_context_ids
        ),
    )


def to_deepeval_test_case(case: JudgeCase):
    """Convert a shared case to a DeepEval LLMTestCase."""
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as error:
        raise RuntimeError(
            "Install judge dependencies with "
            '`python -m pip install -e ".[judge]"`.'
        ) from error

    return LLMTestCase(
        input=case.question,
        actual_output=case.actual_output,
        expected_output=case.expected_output,
        retrieval_context=list(case.retrieval_context),
        context=list(case.reference_context),
    )