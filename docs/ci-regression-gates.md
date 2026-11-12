# CI RAG Regression Gates

The GitHub Actions workflow evaluates the current RAG implementation whenever relevant code, configuration, prompts, retrieval logic, or evaluation assets change.

Unlike a unit-test-only pipeline, this workflow runs the current retrieval and controlled LangGraph workflow against the verified benchmark. A pull request fails when an observed metric violates its configured threshold.

## Evaluation scope

The live regression evaluation covers:

- 10 public GitLab handbook sources
- 19 source-grounded corpus passages
- 20 manually verified benchmark questions
- BM25 retrieval
- Controlled answer, refusal, and escalation decisions
- Grounded answer-quality and citation checks

The benchmark contains answerable, unanswerable, out-of-domain, authorization-sensitive, and escalation cases.

## Quality gates

Thresholds are versioned in `config/quality-gates.json`.

| Area | Metric | Rule |
|---|---|---:|
| Dataset | Public sources | >= 10 |
| Dataset | Corpus passages | >= 19 |
| Dataset | Benchmark questions | >= 20 |
| Retrieval | Recall@3 | >= 0.80 |
| Retrieval | Mean reciprocal rank | >= 0.65 |
| Workflow | Decision accuracy | >= 0.95 |
| Answer quality | Answer correctness | >= 0.50 |
| Grounding | Faithfulness | >= 0.95 |
| Grounding | Hallucination rate | <= 0.05 |
| Citations | Citation precision | >= 0.95 |
| Citations | Citation recall | >= 0.95 |
| Safety behavior | Refusal accuracy | >= 1.00 |
| Safety behavior | Escalation accuracy | >= 1.00 |

Both minimum and maximum rules are supported. Higher values are preferred for quality metrics, while lower values are preferred for hallucination rate.

## Local execution

Run the same evaluation used by CI:

```bash
python scripts/check_quality_gates.py
```

The command:

1. Validates the current dataset size.
2. Evaluates BM25 retrieval.
3. Builds the current retrieval and reranking stack.
4. Runs all benchmark questions through the current LangGraph workflow.
5. Evaluates decisions, answers, evidence support, citations, refusals, and escalations.
6. Compares the observed metrics with the versioned thresholds.
7. Writes a machine-readable report.
8. Returns a nonzero exit code if any gate fails.

The generated report is saved at:

```text
artifacts/runs/ci-quality-gate.json
```

## GitHub Actions behavior

The workflow is defined in:

```text
.github/workflows/evaluation-gates.yml
```

It runs for pull requests and relevant updates to the main branch. The workflow validates the dataset, runs the unit-test suite, and applies the live RAG regression gates.

A failed gate prevents the configuration from being treated as production-ready until the regression is investigated or the implementation is corrected.

## Threshold policy

Thresholds should not be lowered only to make a failing build pass. A threshold change should include:

- A documented reason
- Supporting evaluation evidence
- Review of the affected benchmark cases
- A clear explanation in the pull request

Because the current benchmark is a small verified starter set, passing every gate demonstrates regression safety for the evaluated scope. It does not, by itself, establish full enterprise production readiness.

