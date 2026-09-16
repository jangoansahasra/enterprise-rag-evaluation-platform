"""Command-line entry point for reproducible baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_benchmark, load_corpus
from .evaluation import evaluate_retriever
from .retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/bm25-baseline.json"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    items = load_benchmark(args.benchmark)
    report = evaluate_retriever(
        BM25Retriever(load_corpus(args.corpus)), items, top_k=args.top_k
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
