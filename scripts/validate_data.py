"""Validate the source registry and manually curated benchmark."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sources = json.loads((ROOT / "data/sources.json").read_text())
questions = json.loads((ROOT / "data/benchmark.json").read_text())
corpus = json.loads((ROOT / "data/corpus.json").read_text())
source_by_id = {source["id"]: source for source in sources}
chunk_by_id = {chunk["id"]: chunk for chunk in corpus}

assert len(source_by_id) == len(sources), "Duplicate source ID"
assert len({item["id"] for item in questions}) == len(questions), "Duplicate question ID"
assert len(chunk_by_id) == len(corpus), "Duplicate corpus chunk ID"

for chunk in corpus:
    assert chunk["source_id"] in source_by_id, chunk["id"]
    assert chunk["section"] in source_by_id[chunk["source_id"]]["sections"], chunk["id"]
    assert chunk["text"].strip(), chunk["id"]

for item in questions:
    assert item["expected_behavior"] in {"answer", "refuse", "escalate"}, item["id"]
    assert item["answerable"] == (item["expected_behavior"] == "answer"), item["id"]
    assert bool(item["verified_answer"]) == item["answerable"], item["id"]
    assert item["evidence"] or not item["answerable"], item["id"]
    for evidence in item["evidence"]:
        assert evidence["source_id"] in source_by_id, item["id"]
        assert evidence["section"] in source_by_id[evidence["source_id"]]["sections"], item["id"]
        assert evidence["supporting_passage"].strip(), item["id"]
        assert evidence["source_id"] + "::" + evidence["section"] in chunk_by_id, item["id"]

print(
    f"Validated {len(sources)} sources, {len(corpus)} corpus passages, "
    f"and {len(questions)} benchmark questions."
)
