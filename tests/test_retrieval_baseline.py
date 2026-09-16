from pathlib import Path

from rag_reliability.data import load_benchmark, load_corpus
from rag_reliability.evaluation import evaluate_retriever
from rag_reliability.retrieval import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]


def build_retriever() -> BM25Retriever:
    return BM25Retriever(load_corpus(ROOT / "data/corpus.json"))


def test_bm25_baseline_recovers_verified_evidence() -> None:
    items = load_benchmark(ROOT / "data/benchmark.json")
    report = evaluate_retriever(build_retriever(), items, top_k=3)

    assert report["summary"]["evaluated_questions"] == 16
    assert report["summary"]["recall_at_3"] >= 0.80


def test_unanswerable_questions_are_excluded_from_retrieval_recall() -> None:
    items = load_benchmark(ROOT / "data/benchmark.json")
    report = evaluate_retriever(build_retriever(), items, top_k=3)

    evaluated_ids = {item["question_id"] for item in report["questions"]}
    assert {"q006", "q007", "q008", "q010"}.isdisjoint(evaluated_ids)
