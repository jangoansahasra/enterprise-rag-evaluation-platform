"""Tests for versioned document processing and chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_reliability.document_processing import (
    build_chunk_manifest,
    chunk_documents,
    fixed_size_split,
    load_source_documents,
    recursive_split,
)


def write_dataset(
    directory: Path,
    *,
    passage_text: str = (
        "Use the approved password manager for company accounts."
    ),
    source_id: str = "policy-passwords",
    passage_source_id: str | None = None,
    registered_sections: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write a minimal source registry and independent corpus."""
    sources_path = directory / "sources.json"
    corpus_path = directory / "corpus.json"

    sources = [
        {
            "id": source_id,
            "title": "Password Policy",
            "url": "https://example.com/password-policy",
            "publisher": "Example Organization",
            "license": "CC BY 4.0",
            "checked_date": "2026-11-03",
            "sections": (
                registered_sections
                if registered_sections is not None
                else ["Approved tools"]
            ),
        }
    ]
    corpus = [
        {
            "id": f"{source_id}::Approved tools",
            "source_id": passage_source_id or source_id,
            "section": "Approved tools",
            "text": passage_text,
        }
    ]

    sources_path.write_text(
        json.dumps(sources),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps(corpus),
        encoding="utf-8",
    )
    return sources_path, corpus_path


def test_document_versions_are_stable_and_content_sensitive(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(tmp_path)

    first = load_source_documents(
        sources_path,
        corpus_path,
    )
    second = load_source_documents(
        sources_path,
        corpus_path,
    )

    assert first[0].version == second[0].version
    assert len(first[0].version) == 16

    write_dataset(
        tmp_path,
        passage_text=(
            "Use the approved password manager and enable MFA."
        ),
    )
    changed = load_source_documents(
        sources_path,
        corpus_path,
    )

    assert changed[0].version != first[0].version


def test_loader_preserves_source_and_section_metadata(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(tmp_path)

    documents = load_source_documents(
        sources_path,
        corpus_path,
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "policy-passwords"
    assert document.title == "Password Policy"
    assert document.publisher == "Example Organization"
    assert document.license == "CC BY 4.0"
    assert document.checked_date == "2026-11-03"
    assert document.sections[0].heading == "Approved tools"
    assert document.sections[0].id == (
        "policy-passwords::Approved tools"
    )


def test_loader_rejects_unknown_source_ids(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(
        tmp_path,
        passage_source_id="unknown-source",
    )

    with pytest.raises(
        ValueError,
        match="unknown source IDs",
    ):
        load_source_documents(
            sources_path,
            corpus_path,
        )


def test_loader_rejects_unregistered_sections(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(
        tmp_path,
        registered_sections=["Different section"],
    )

    with pytest.raises(
        ValueError,
        match="unregistered sections",
    ):
        load_source_documents(
            sources_path,
            corpus_path,
        )


def test_fixed_size_split_applies_overlap() -> None:
    chunks = fixed_size_split(
        "abcdefghij",
        chunk_size=6,
        overlap=2,
    )

    assert chunks == [
        "abcdef",
        "efghij",
    ]


def test_recursive_split_prefers_natural_boundaries() -> None:
    chunks = recursive_split(
        "Alpha beta. Gamma delta. Epsilon zeta.",
        chunk_size=25,
        overlap=0,
    )

    assert chunks == [
        "Alpha beta. Gamma delta.",
        "Epsilon zeta.",
    ]
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_invalid_chunk_parameters_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_size",
    ):
        fixed_size_split(
            "text",
            chunk_size=0,
            overlap=0,
        )

    with pytest.raises(
        ValueError,
        match="smaller than",
    ):
        recursive_split(
            "text",
            chunk_size=10,
            overlap=10,
        )


def test_heading_aware_chunks_preserve_provenance(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(tmp_path)
    documents = load_source_documents(
        sources_path,
        corpus_path,
    )

    chunks = chunk_documents(
        documents,
        strategy="heading_aware",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.parent_id == (
        "policy-passwords::Approved tools"
    )
    assert chunk.source_id == "policy-passwords"
    assert chunk.section == "Approved tools"
    assert chunk.document_title == "Password Policy"
    assert chunk.source_url == (
        "https://example.com/password-policy"
    )
    assert chunk.strategy == "heading_aware"
    assert chunk.chunk_size == 0
    assert chunk.overlap == 0
    assert chunk.document_version == documents[0].version


def test_fixed_and_recursive_chunks_have_stable_ids(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(
        tmp_path,
        passage_text=(
            "First sentence. Second sentence. Third sentence."
        ),
    )
    documents = load_source_documents(
        sources_path,
        corpus_path,
    )

    fixed_chunks = chunk_documents(
        documents,
        strategy="fixed_size",
        chunk_size=20,
        overlap=5,
    )
    recursive_chunks = chunk_documents(
        documents,
        strategy="recursive",
        chunk_size=20,
        overlap=5,
    )

    assert fixed_chunks[0].id.endswith(
        "::fixed-20-5::000"
    )
    assert recursive_chunks[0].id.endswith(
        "::recursive-20-5::000"
    )
    assert all(
        chunk.parent_id
        == "policy-passwords::Approved tools"
        for chunk in fixed_chunks + recursive_chunks
    )


def test_manifest_records_configuration_and_versions(
    tmp_path: Path,
) -> None:
    sources_path, corpus_path = write_dataset(tmp_path)
    documents = load_source_documents(
        sources_path,
        corpus_path,
    )
    chunks = chunk_documents(
        documents,
        strategy="fixed_size",
        chunk_size=30,
        overlap=5,
    )

    manifest = build_chunk_manifest(chunks)

    assert manifest["strategy"] == "fixed_size"
    assert manifest["chunk_size"] == 30
    assert manifest["overlap"] == 5
    assert manifest["source_count"] == 1
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["document_versions"] == {
        "policy-passwords": documents[0].version
    }