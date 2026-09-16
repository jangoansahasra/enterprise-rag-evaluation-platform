"""Typed loaders for the source registry and gold benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Evidence:
    source_id: str
    section: str
    supporting_passage: str

    @property
    def chunk_id(self) -> str:
        return f"{self.source_id}::{self.section}"


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    question: str
    category: str
    difficulty: str
    answerable: bool
    expected_behavior: str
    verified_answer: str | None
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    section: str
    text: str


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        BenchmarkItem(
            id=item["id"],
            question=item["question"],
            category=item["category"],
            difficulty=item["difficulty"],
            answerable=item["answerable"],
            expected_behavior=item["expected_behavior"],
            verified_answer=item["verified_answer"],
            evidence=tuple(Evidence(**evidence) for evidence in item["evidence"]),
        )
        for item in raw
    ]


def load_corpus(path: Path) -> list[Chunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


def evidence_chunks(items: list[BenchmarkItem]) -> list[Chunk]:
    """Build chunks from labels for data inspection only, never retrieval evaluation."""
    chunks: dict[str, Chunk] = {}
    for item in items:
        for evidence in item.evidence:
            existing = chunks.get(evidence.chunk_id)
            text = evidence.supporting_passage
            if existing and text not in existing.text:
                text = f"{existing.text} {text}"
            chunks[evidence.chunk_id] = Chunk(
                id=evidence.chunk_id,
                source_id=evidence.source_id,
                section=evidence.section,
                text=text,
            )
    return sorted(chunks.values(), key=lambda chunk: chunk.id)
