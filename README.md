# Enterprise RAG Reliability & Evaluation Platform

A source-grounded evaluation platform for an enterprise IT support assistant. This project starts with real public GitLab handbook pages and a manually verified benchmark. No synthetic policies or model-generated ground truth are used.

## Current stage

Milestone 1 is underway: source selection, provenance, and benchmark design. The audited starter release contains 10 public handbook sources and 20 benchmark questions across direct, reasoning, ambiguous, multi-document, adversarial, unanswerable, and out-of-domain cases. No GitHub repository has been created.

Run `python3 scripts/validate_data.py` to check the starter data. The source manifest records provenance, `corpus.json` contains the independently loaded searchable passages, and `benchmark.json` contains gold labels. Keeping the corpus separate from the benchmark prevents evaluation labels from being used to construct the retriever.

## Reproducible baseline

The first working slice is a dependency-free BM25 retrieval baseline. It intentionally scores retrieval before adding answer generation so retrieval failures and generation failures remain distinguishable.

```bash
PYTHONPATH=src python3 -m rag_reliability.cli
```

The command writes `artifacts/runs/bm25-baseline.json` with Recall@3, mean reciprocal rank, and per-question rankings. Frozen comparison results are stored in `artifacts/baselines/`; `bm25-v0.2.json` evaluates the expanded starter benchmark. See [the architecture](docs/architecture.md) and [delivery plan](docs/project-plan.md) for how this grows into the full reliability platform.

## Data rules

- Keep the handbook's organization and policy context explicit. Do not present GitLab policy as general advice for other employers.
- Record the source URL and section for every answerable question.
- Use `escalate` when a request requires a human decision or authorization.
- Use `refuse` for requests that ask the assistant to bypass a documented restriction.
- Recheck source pages before a released benchmark version; public handbook content can change.
- Credit GitLab and comply with its [handbook reuse terms](https://handbook.gitlab.com/handbook/about/handbook-usage/).

## Next build steps

1. Expand and freeze the source corpus with version identifiers and local licensed snapshots.
2. Review the benchmark manually, including ambiguous and multi-document cases.
3. Implement document ingestion and BM25 retrieval as the first measured baseline.
