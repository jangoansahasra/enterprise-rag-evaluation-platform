# Answer-quality evaluation

The controlled RAG workflow is evaluated against the source-verified 20-question benchmark without requiring hosted-model credentials.

## Metrics

The evaluation records:

- **Decision accuracy:** whether the workflow correctly answers, refuses, or escalates.
- **Answer correctness:** normalized lexical token F1 against the verified answer.
- **Faithfulness:** proportion of generated answer terms supported by retrieved evidence.
- **Hallucination rate:** one minus the deterministic faithfulness score.
- **Citation precision:** proportion of returned citations that match verified supporting sources.
- **Citation recall:** proportion of verified supporting sources included in the response.
- **Refusal accuracy:** correctness on questions expected to be refused.
- **Escalation accuracy:** correctness on questions requiring human authorization.

## Baseline results

| Metric | Score |
|---|---:|
| Decision accuracy | 1.0000 |
| Answer correctness | 0.5618 |
| Faithfulness | 1.0000 |
| Hallucination rate | 0.0000 |
| Citation precision | 0.9375 |
| Citation recall | 0.9062 |
| Refusal accuracy | 1.0000 |
| Escalation accuracy | 1.0000 |

## Findings

The current workflow is fully faithful under the deterministic metric because answerable responses quote retrieved evidence directly. This design limits unsupported generation but produces less concise answers than the manually verified references, resulting in an answer-correctness score of 0.5618.

Two citation weaknesses remain:

- `q015` retrieves and cites the general suspicious-message procedure instead of the Gmail-specific PhishArm procedure.
- `q019` correctly cites the AI support data-handling policy but includes only one of the two verified sources required for the multi-document answer.

These cases demonstrate why decision accuracy alone is insufficient. A workflow can choose the correct high-level behavior while still return incomplete or less relevant evidence.

## Limitations

The credential-free metrics are deterministic proxies. Lexical overlap does not fully measure semantic equivalence, and term coverage does not perform natural-language inference. The report therefore distinguishes these metrics from future RAGAS and DeepEval judge-based results.

The starter benchmark contains only 20 curated questions and is not sufficient to establish production readiness.

## Run locally

~~~bash
PYTHONPATH=src python scripts/evaluate_answer_quality.py
~~~

The generated report is written to:

~~~text
artifacts/runs/answer-quality-evaluation.json
~~~

The versioned baseline is stored at:

~~~text
artifacts/baselines/answer-quality-v0.1.json
~~~