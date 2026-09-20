"""Controlled LangGraph workflow for evidence-based support decisions."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .retrieval import SearchResult, tokenize


Decision = Literal["answer", "refuse", "escalate"]
Classification = Literal["in_domain", "out_of_domain", "authorization"]
EvidenceQuality = Literal["sufficient", "insufficient"]


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return ranked evidence for a question."""


class RetrievedEvidence(TypedDict):
    chunk_id: str
    source_id: str
    section: str
    text: str
    score: float


class WorkflowState(TypedDict, total=False):
    question: str
    classification: Classification
    retrieved_evidence: list[RetrievedEvidence]
    evidence_overlap: float
    evidence_quality: EvidenceQuality
    required_claim_terms: list[str]
    unsupported_claim_terms: list[str]
    decision: Decision
    answer: str
    citations: list[str]
    reason: str


DOMAIN_TERMS = {
    "access",
    "account",
    "application",
    "credential",
    "customer",
    "data",
    "device",
    "email",
    "enroll",
    "gitlab",
    "incident",
    "it",
    "jamf",
    "laptop",
    "manager",
    "message",
    "okta",
    "password",
    "phishing",
    "policy",
    "security",
    "sla",
    "social",
    "software",
    "suspicious",
    "support",
    "usb",
    "vpn",
}

AUTHORIZATION_TERMS = {
    "approve",
    "authorize",
    "bypass",
    "disable",
    "grant",
    "permanently",
}


SPECIFICITY_TERMS = {
    "guaranteed",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "at",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "if",
    "in",
    "initially",
    "into",
    "is",
    "it",
    "may",
    "me",
    "my",
    "of",
    "should",
    "the",
    "to",
    "what",
    "which",
    "who",
    "with",
}


def normalize_term(token: str) -> str:
    """Apply small deterministic reductions for retrieval-overlap checks."""
    if token.endswith("ss"):
        return token

    for suffix in ("ions", "ing", "ed", "ion", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]

    return token


def meaningful_terms(text: str) -> set[str]:
    return {
        normalize_term(token)
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) > 1
    }


def build_workflow(
    retriever: Retriever,
    *,
    top_k: int = 3,
    evidence_threshold: float = 0.20,
):
    """Build a bounded workflow with no autonomous tool-selection loop."""

    def classify_question(state: WorkflowState) -> WorkflowState:
        question_terms = meaningful_terms(state["question"])

        if question_terms & AUTHORIZATION_TERMS:
            classification: Classification = "authorization"
        elif question_terms & DOMAIN_TERMS:
            classification = "in_domain"
        else:
            classification = "out_of_domain"

        return {"classification": classification}

    def route_question(state: WorkflowState) -> str:
        return state["classification"]

    def retrieve_evidence(state: WorkflowState) -> WorkflowState:
        results = retriever.search(state["question"], top_k=top_k)

        evidence: list[RetrievedEvidence] = [
            {
                "chunk_id": result.chunk.id,
                "source_id": result.chunk.source_id,
                "section": result.chunk.section,
                "text": result.chunk.text,
                "score": result.score,
            }
            for result in results
        ]

        return {"retrieved_evidence": evidence}

    def assess_evidence(state: WorkflowState) -> WorkflowState:
        question_terms = meaningful_terms(state["question"])
        evidence = state.get("retrieved_evidence", [])

        evidence_term_sets = [
            meaningful_terms(f"{item['section']} {item['text']}")
            for item in evidence
        ]
        overlaps = [
            len(question_terms & terms) / len(question_terms)
            for terms in evidence_term_sets
            if question_terms
        ]
        overlap = max(overlaps, default=0.0)

        required_claim_terms = question_terms & SPECIFICITY_TERMS
        top_evidence_terms = evidence_term_sets[0] if evidence_term_sets else set()
        unsupported_claim_terms = required_claim_terms - top_evidence_terms

        quality: EvidenceQuality = (
            "sufficient"
            if overlap >= evidence_threshold and not unsupported_claim_terms
            else "insufficient"
        )

        return {
            "evidence_overlap": overlap,
            "evidence_quality": quality,
            "required_claim_terms": sorted(required_claim_terms),
            "unsupported_claim_terms": sorted(unsupported_claim_terms),
        }

    def route_evidence(state: WorkflowState) -> str:
        return state["evidence_quality"]

    def answer_with_evidence(state: WorkflowState) -> WorkflowState:
        evidence = state["retrieved_evidence"][0]
        citation = evidence["chunk_id"]

        return {
            "decision": "answer",
            "answer": evidence["text"],
            "citations": [citation],
            "reason": "Evidence passed the deterministic support threshold.",
        }

    def refuse_out_of_domain(state: WorkflowState) -> WorkflowState:
        del state
        return {
            "decision": "refuse",
            "answer": (
                "I can only answer questions covered by the enterprise IT "
                "and security knowledge base."
            ),
            "citations": [],
            "reason": "The question was classified as out of domain.",
        }

    def escalate_authorization(state: WorkflowState) -> WorkflowState:
        del state
        return {
            "decision": "escalate",
            "answer": (
                "This request requires authorization and must be reviewed "
                "by the responsible team."
            ),
            "citations": [],
            "reason": "The request requires human authorization.",
        }

    def refuse_unsupported_claim(state: WorkflowState) -> WorkflowState:
        del state
        return {
            "decision": "refuse",
            "answer": (
                "I could not find evidence supporting the requested claim "
                "in the knowledge base."
            ),
            "citations": [],
            "reason": "Retrieved evidence did not support the specific claim.",
        }

    builder = StateGraph(WorkflowState)

    builder.add_node("classify_question", classify_question)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("assess_evidence", assess_evidence)
    builder.add_node("answer_with_evidence", answer_with_evidence)
    builder.add_node("refuse_out_of_domain", refuse_out_of_domain)
    builder.add_node("escalate_authorization", escalate_authorization)
    builder.add_node("refuse_unsupported_claim", refuse_unsupported_claim)

    builder.add_edge(START, "classify_question")
    builder.add_conditional_edges(
        "classify_question",
        route_question,
        {
            "in_domain": "retrieve_evidence",
            "out_of_domain": "refuse_out_of_domain",
            "authorization": "escalate_authorization",
        },
    )
    builder.add_edge("retrieve_evidence", "assess_evidence")
    builder.add_conditional_edges(
        "assess_evidence",
        route_evidence,
        {
            "sufficient": "answer_with_evidence",
            "insufficient": "refuse_unsupported_claim",
        },
    )
    builder.add_edge("answer_with_evidence", END)
    builder.add_edge("refuse_out_of_domain", END)
    builder.add_edge("escalate_authorization", END)
    builder.add_edge("refuse_unsupported_claim", END)

    return builder.compile()
