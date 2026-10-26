"""Credential-free metrics for evaluating generated RAG answers."""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable

from rag_reliability.data import BenchmarkItem
from rag_reliability.workflow import meaningful_terms


def safe_mean(values: Iterable[float | None]) -> float | None:
    """Calculate a mean while ignoring unavailable metrics."""
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def token_f1(candidate: str, reference: str) -> float:
    """Measure lexical overlap between an answer and its reference."""
    candidate_terms = meaningful_terms(candidate)
    reference_terms = meaningful_terms(reference)

    if not candidate_terms or not reference_terms:
        return 0.0

    shared = candidate_terms & reference_terms
    precision = len(shared) / len(candidate_terms)
    recall = len(shared) / len(reference_terms)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def evidence_support(
    answer: str,
    contexts: list[str],
) -> float:
    """Measure how much of an answer is supported by retrieved context."""
    answer_terms = meaningful_terms(answer)
    context_terms = meaningful_terms(" ".join(contexts))

    if not answer_terms:
        return 1.0

    return len(answer_terms & context_terms) / len(answer_terms)


def citation_scores(
    actual_citations: list[str],
    expected_citations: list[str],
) -> tuple[float | None, float | None]:
    """Calculate citation precision and recall."""
    actual = set(actual_citations)
    expected = set(expected_citations)

    if not actual and not expected:
        return None, None

    overlap = actual & expected
    precision = len(overlap) / len(actual) if actual else 0.0
    recall = len(overlap) / len(expected) if expected else 0.0
    return precision, recall


def evaluate_answer(
    benchmark_item: BenchmarkItem,
    workflow_output: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one workflow response against its verified label."""
    expected_decision = benchmark_item.expected_behavior
    actual_decision = workflow_output["decision"]
    decision_correct = actual_decision == expected_decision

    retrieved = workflow_output.get(
        "retrieved_evidence",
        workflow_output.get("retrieved", []),
    )
    contexts = [
        item["text"]
        if isinstance(item, dict)
        else item.chunk.text
        for item in retrieved
    ]
    expected_citations = [
        evidence.chunk_id for evidence in benchmark_item.evidence
    ]
    actual_citations = workflow_output.get("citations", [])

    answer_correctness: float | None = None
    faithfulness: float | None = None
    hallucination_rate: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None

    if actual_decision == "answer":
        faithfulness = evidence_support(
            workflow_output["answer"],
            contexts,
        )
        hallucination_rate = 1.0 - faithfulness
        citation_precision, citation_recall = citation_scores(
            actual_citations,
            expected_citations,
        )

        if benchmark_item.verified_answer:
            answer_correctness = token_f1(
                workflow_output["answer"],
                benchmark_item.verified_answer,
            )

    return {
        "question_id": benchmark_item.id,
        "category": benchmark_item.category,
        "difficulty": benchmark_item.difficulty,
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
        "decision_correct": decision_correct,
        "answer_correctness": answer_correctness,
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "refusal_correct": (
            decision_correct if expected_decision == "refuse" else None
        ),
        "escalation_correct": (
            decision_correct
            if expected_decision == "escalate"
            else None
        ),
        "answer": workflow_output["answer"],
        "citations": actual_citations,
        "expected_citations": expected_citations,
        "reason": workflow_output["reason"],
    }


def summarize_answer_evaluations(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-question answer-quality results."""
    question_count = len(cases)
    correct_decisions = sum(
        bool(case["decision_correct"]) for case in cases
    )

    return {
        "evaluated_questions": question_count,
        "decision_accuracy": (
            correct_decisions / question_count
            if question_count
            else 0.0
        ),
        "answer_correctness": safe_mean(
            case["answer_correctness"] for case in cases
        ),
        "faithfulness": safe_mean(
            case["faithfulness"] for case in cases
        ),
        "hallucination_rate": safe_mean(
            case["hallucination_rate"] for case in cases
        ),
        "citation_precision": safe_mean(
            case["citation_precision"] for case in cases
        ),
        "citation_recall": safe_mean(
            case["citation_recall"] for case in cases
        ),
        "refusal_accuracy": safe_mean(
            float(case["refusal_correct"])
            if case["refusal_correct"] is not None
            else None
            for case in cases
        ),
        "escalation_accuracy": safe_mean(
            float(case["escalation_correct"])
            if case["escalation_correct"] is not None
            else None
            for case in cases
        ),
    }