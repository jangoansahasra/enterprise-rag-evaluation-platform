# Recruiter-focused delivery plan

Each milestone ends with a runnable result and evidence of what improved. Features are added only when they support a comparison or a reliability decision.

## 1. Gold dataset and lexical baseline — in progress

- Expand the corpus to 15–25 coherent public GitLab IT and security pages.
- Expand the benchmark to at least 100 manually reviewed questions.
- Freeze source versions and record attribution, URL, retrieval date, and content hash.
- Report Recall@k, MRR, latency, and failures for BM25.

Exit criteria: two-person-style review checklist completed, no orphaned citations, and category coverage documented.

## 2. Retrieval experiments

- Implement fixed, recursive, and heading-aware chunking first; add semantic chunking only if it answers a measured weakness.
- Compare BM25, sentence-transformer dense retrieval, reciprocal-rank hybrid retrieval, and cross-encoder reranking.
- Keep a shared experiment contract and write every result to versioned JSON or Parquet.

Exit criteria: a selected retrieval configuration with documented quality and latency tradeoffs.

## 3. Controlled answer workflow

- Add LangGraph nodes for domain classification, retrieval, evidence thresholding, one bounded rewrite attempt, answer generation, groundedness verification, and escalation.
- Require structured responses with citation IDs and a decision of `answer`, `refuse`, or `escalate`.
- Support a local model or recorded-response demo; hosted providers remain optional.

Exit criteria: every state transition is testable, traceable, and bounded.

## 4. Evaluation and observability

- Add deterministic correctness, citation, refusal, escalation, latency, and cost metrics.
- Add RAGAS and DeepEval judges as labeled secondary signals.
- Track configurations and artifacts in MLflow and emit OpenTelemetry-compatible traces.

Exit criteria: a failed score can be traced to a question, retrieved chunks, model output, and configuration.

## 5. Reliability dashboard and CI gates

- Build a Streamlit dashboard around saved results so the portfolio works without credentials.
- Show comparisons, category slices, regressions, cost/latency tradeoffs, and individual failures.
- Add a small deterministic CI suite plus threshold and critical-case regression gates.

Exit criteria: a reviewer can identify the best configuration and explain why another candidate failed.

## 6. Portfolio release

- Add Docker, GitHub Actions, system/model cards, architecture diagrams, screenshots, and a short demo recording.
- Document limitations, dataset provenance, evaluation bias, and reproducibility.

Exit criteria: a fresh clone launches the saved-results dashboard and can reproduce the free local baseline.
