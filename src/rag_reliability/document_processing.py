"""Versioned document loading and metadata-preserving chunking."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


ChunkingStrategy = Literal[
    "heading_aware",
    "fixed_size",
    "recursive",
]


@dataclass(frozen=True)
class DocumentSection:
    """One independently collected section from a public source."""

    id: str
    heading: str
    text: str


@dataclass(frozen=True)
class SourceDocument:
    """A versioned public document with preserved source metadata."""

    id: str
    title: str
    url: str
    publisher: str
    license: str
    checked_date: str
    version: str
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True)
class ProcessedChunk:
    """A searchable chunk linked to its original source passage."""

    id: str
    source_id: str
    section: str
    text: str
    parent_id: str
    document_title: str
    source_url: str
    publisher: str
    license: str
    checked_date: str
    document_version: str
    strategy: str
    chunk_size: int
    overlap: int
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing the source wording."""
    return re.sub(r"\s+", " ", text).strip()


def document_version(
    source: dict[str, Any],
    sections: list[dict[str, str]],
) -> str:
    """Create a stable content hash for one source document."""
    version_input = {
        "source_id": source["id"],
        "title": source["title"],
        "url": source["url"],
        "publisher": source["publisher"],
        "license": source["license"],
        "checked_date": source["checked_date"],
        "sections": [
            {
                "id": section["id"],
                "heading": section["section"],
                "text": normalize_text(section["text"]),
            }
            for section in sections
        ],
    }
    canonical = json.dumps(
        version_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:16]


def load_source_documents(
    sources_path: Path,
    corpus_path: Path,
) -> list[SourceDocument]:
    """Load public source metadata and independently collected passages."""
    source_records = json.loads(
        sources_path.read_text(encoding="utf-8")
    )
    corpus_records = json.loads(
        corpus_path.read_text(encoding="utf-8")
    )

    corpus_by_source: dict[str, list[dict[str, str]]] = {}
    for passage in corpus_records:
        corpus_by_source.setdefault(
            passage["source_id"],
            [],
        ).append(passage)

    known_source_ids = {
        source["id"] for source in source_records
    }
    unknown_source_ids = (
        set(corpus_by_source) - known_source_ids
    )
    if unknown_source_ids:
        raise ValueError(
            "Corpus contains unknown source IDs: "
            + ", ".join(sorted(unknown_source_ids))
        )

    documents = []

    for source in source_records:
        passages = corpus_by_source.get(source["id"], [])
        registered_sections = set(source["sections"])

        unknown_sections = {
            passage["section"]
            for passage in passages
            if passage["section"] not in registered_sections
        }
        if unknown_sections:
            raise ValueError(
                f"Source {source['id']} contains unregistered "
                f"sections: {', '.join(sorted(unknown_sections))}"
            )

        sections = tuple(
            DocumentSection(
                id=passage["id"],
                heading=passage["section"],
                text=normalize_text(passage["text"]),
            )
            for passage in passages
        )

        documents.append(
            SourceDocument(
                id=source["id"],
                title=source["title"],
                url=source["url"],
                publisher=source["publisher"],
                license=source["license"],
                checked_date=source["checked_date"],
                version=document_version(
                    source,
                    passages,
                ),
                sections=sections,
            )
        )

    return documents


def fixed_size_split(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into deterministic character windows."""
    _validate_chunk_parameters(chunk_size, overlap)
    normalized = normalize_text(text)

    if not normalized:
        return []

    chunks = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(normalized):
            break

        start = end - overlap

    return chunks


def recursive_split(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Prefer natural boundaries while respecting a maximum size."""
    _validate_chunk_parameters(chunk_size, overlap)
    normalized = normalize_text(text)

    if not normalized:
        return []

    separators = (". ", "; ", ": ", ", ", " ")
    chunks = []
    start = 0

    while start < len(normalized):
        maximum_end = min(
            start + chunk_size,
            len(normalized),
        )
        end = maximum_end

        if maximum_end < len(normalized):
            window = normalized[start:maximum_end]
            minimum_length = max(
                1,
                chunk_size // 2,
            )
            boundary = -1
            boundary_width = 0

            for separator in separators:
                position = window.rfind(separator)

                if (
                    position >= minimum_length
                    and position > boundary
                ):
                    boundary = position
                    boundary_width = len(separator)

            if boundary >= 0:
                end = start + boundary + boundary_width

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(normalized):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    *,
    strategy: ChunkingStrategy,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[ProcessedChunk]:
    """Create searchable chunks while preserving provenance."""
    if strategy not in {
        "heading_aware",
        "fixed_size",
        "recursive",
    }:
        raise ValueError(
            f"Unsupported chunking strategy: {strategy}"
        )

    if strategy == "heading_aware":
        effective_size = 0
        effective_overlap = 0
    else:
        _validate_chunk_parameters(chunk_size, overlap)
        effective_size = chunk_size
        effective_overlap = overlap

    chunks = []

    for document in documents:
        for section in document.sections:
            if strategy == "heading_aware":
                pieces = [section.text]
                configuration = "heading"
            elif strategy == "fixed_size":
                pieces = fixed_size_split(
                    section.text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                configuration = (
                    f"fixed-{chunk_size}-{overlap}"
                )
            else:
                pieces = recursive_split(
                    section.text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                configuration = (
                    f"recursive-{chunk_size}-{overlap}"
                )

            for index, piece in enumerate(pieces):
                chunk_id = (
                    f"{section.id}::{configuration}::{index:03d}"
                )
                chunks.append(
                    ProcessedChunk(
                        id=chunk_id,
                        source_id=document.id,
                        section=section.heading,
                        text=piece,
                        parent_id=section.id,
                        document_title=document.title,
                        source_url=document.url,
                        publisher=document.publisher,
                        license=document.license,
                        checked_date=document.checked_date,
                        document_version=document.version,
                        strategy=strategy,
                        chunk_size=effective_size,
                        overlap=effective_overlap,
                        chunk_index=index,
                    )
                )

    return chunks


def build_chunk_manifest(
    chunks: list[ProcessedChunk],
) -> dict[str, Any]:
    """Summarize one reproducible chunking configuration."""
    source_ids = {
        chunk.source_id for chunk in chunks
    }
    versions = {
        chunk.source_id: chunk.document_version
        for chunk in chunks
    }
    strategies = {
        chunk.strategy for chunk in chunks
    }
    sizes = {
        chunk.chunk_size for chunk in chunks
    }
    overlaps = {
        chunk.overlap for chunk in chunks
    }

    if len(strategies) > 1:
        raise ValueError(
            "A manifest must contain one chunking strategy."
        )
    if len(sizes) > 1 or len(overlaps) > 1:
        raise ValueError(
            "A manifest must contain one chunk-size configuration."
        )

    return {
        "strategy": next(iter(strategies), None),
        "chunk_size": next(iter(sizes), None),
        "overlap": next(iter(overlaps), None),
        "source_count": len(source_ids),
        "chunk_count": len(chunks),
        "document_versions": dict(sorted(versions.items())),
    }


def _validate_chunk_parameters(
    chunk_size: int,
    overlap: int,
) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )