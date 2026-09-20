"""Dense retrieval using Sentence Transformers and FAISS."""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .data import Chunk
from .retrieval import SearchResult


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if not chunks:
            raise ValueError("Dense retrieval requires at least one chunk")

        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        embeddings = self.model.encode(
            [f"{chunk.section}\n{chunk.text}" for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        result_count = min(top_k, len(self.chunks))

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        scores, indices = self.index.search(query_embedding, result_count)

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(score),
            )
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]