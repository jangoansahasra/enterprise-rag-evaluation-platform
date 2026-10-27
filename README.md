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

Current retrieval findings and limitations are documented in [the retrieval experiment report](docs/retrieval-results.md).

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

## Controlled decision workflow

The bounded answer, refusal, and escalation flow is documented in the [controlled workflow report](docs/controlled-workflow.md).

## Reliability dashboard

The Streamlit dashboard reads versioned evaluation artifacts, so it works without hosted-model credentials.

```bash
python -m streamlit run dashboard/app.py
```
It presents retrieval quality, latency tradeoffs, workflow decisions, failed cases, CI thresholds, and the current production-readiness assessment.

Stop Streamlit with `Ctrl + C`, then verify:

```bash
python -m py_compile dashboard/app.py
python -m pytest -q
git diff --check
git status
```

## Experiment tracking and observability

Retrieval experiments create separate MLflow runs for BM25, dense, hybrid, and reranked configurations. Each run records configuration parameters, retrieval metrics, latency, tags, and the complete evaluation report.

Structured JSON logs and OpenTelemetry spans capture request IDs, configuration details, evaluation results, latency, failures, and trace context.

```bash
PYTHONPATH=src python scripts/run_retrieval_experiments.py

python -m mlflow ui \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --default-artifact-root ./artifacts/mlflow/artifacts \
  --port 5000
  ```

  ## Answer-quality evaluation

The platform evaluates answer correctness, evidence faithfulness, hallucination rate, citation precision and recall, refusal correctness, and escalation correctness. The first credential-free baseline achieved 100% decision accuracy and deterministic faithfulness, while identifying two citation weaknesses and an answer-conciseness gap.

```bash
PYTHONPATH=src python scripts/evaluate_answer_quality.py
```
See [answer-quality evaluation](docs/answer-quality-evaluation.md) for metric definitions, results, and limitations.

A regression-driven evidence-selection improvement uses heading-aware lexical coverage with a cross-encoder score guard and bounded multi-document aggregation. It increased citation precision from 93.75% to 100% and citation recall from 90.62% to 100% without reducing decision accuracy or faithfulness. See [evidence-selection improvement](docs/evidence-selection.md).
