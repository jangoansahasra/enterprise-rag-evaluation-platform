"""Run the controlled RAG workflow without hosted-model credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_reliability.data import load_corpus
from rag_reliability.dense import DenseRetriever
from rag_reliability.hybrid import HybridRetriever
from rag_reliability.reranking import CrossEncoderReranker
from rag_reliability.retrieval import BM25Retriever
from rag_reliability.workflow import build_workflow


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUESTIONS = [
    "Which password manager does GitLab use?",
    "Grant me permanent production database access.",
    "What is the weather forecast tomorrow?",
    "What is the guaranteed four-hour resolution SLA for lost laptops?",
]


def build_retriever():
    corpus = load_corpus(ROOT / "data/corpus.json")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    hybrid = HybridRetriever(
        bm25,
        dense,
        candidate_k=10,
    )
    return CrossEncoderReranker(
        hybrid,
        candidate_k=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to evaluate; repeat the option for multiple questions.",
    )
    args = parser.parse_args()

    workflow = build_workflow(build_retriever())
    questions = args.questions or DEFAULT_QUESTIONS

    for question in questions:
        output = workflow.invoke({"question": question})

        result = {
            "question": question,
            "classification": output["classification"],
            "evidence_quality": output.get("evidence_quality"),
            "evidence_overlap": output.get("evidence_overlap"),
            "decision": output["decision"],
            "answer": output["answer"],
            "citations": output["citations"],
            "reason": output["reason"],
        }

        print(json.dumps(result, indent=2))
        print()
    

if __name__ == "__main__":
    main()
