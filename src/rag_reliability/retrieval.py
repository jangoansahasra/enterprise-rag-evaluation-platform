"""Dependency-free BM25 baseline used as the first retrieval benchmark."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .data import Chunk


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(terms.values()) for terms in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency = Counter(
            term for terms in self.term_frequencies for term in terms
        )
        count = len(chunks)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        query_terms = tokenize(query)
        scored: list[SearchResult] = []
        for chunk, frequencies, length in zip(
            self.chunks, self.term_frequencies, self.lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            scored.append(SearchResult(chunk=chunk, score=score))
        scored.sort(key=lambda result: (-result.score, result.chunk.id))
        return scored[:top_k]
