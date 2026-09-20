# Retrieval experiment results

## Starter benchmark

The experiment uses 20 manually verified questions and 19 source-grounded passages from 10 public GitLab handbook pages. Sixteen questions are answerable and included in retrieval scoring.

| Configuration | Recall@3 | MRR | Mean query time |
|---|---:|---:|---:|
| BM25 | 0.8125 | 0.6875 | 0.026 ms |
| Dense retrieval | 0.9375 | 0.8646 | 10.368 ms |
| Hybrid RRF | 0.9375 | 0.8750 | 5.620 ms |
| Hybrid with cross-encoder reranking | 1.0000 | 0.9688 | 18.889 ms |

Dense retrieval substantially improved evidence recall over BM25. Reciprocal-rank fusion improved ranking quality but did not recover the remaining missed question. Cross-encoder reranking recovered all verified evidence within the top three and produced the highest MRR.

The latency figures come from a single local run over a small corpus. They demonstrate the relative cost of each configuration but should not be treated as production latency estimates. Later experiments will use warm-up iterations, repeated runs, percentile latency, and a larger corpus.

The versioned per-question results are stored in `artifacts/baselines/retrieval-comparison-v0.2.json`.
