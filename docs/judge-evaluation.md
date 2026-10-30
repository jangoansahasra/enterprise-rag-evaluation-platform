# RAGAS and DeepEval integration

This project provides reproducible evaluation adapters and credential-free
baseline reports for RAGAS and DeepEval.

The evaluation uses 16 answerable cases derived from the verified benchmark.
Refusal and escalation cases remain covered by the deterministic workflow
evaluation because they do not contain generated answers suitable for standard
answer-quality metrics.

## Shared judge-case format

The judge-case export contains:

- Question and question ID
- Generated answer
- Verified answer
- Retrieved context text
- Verified reference context
- Retrieved passage IDs
- Verified reference passage IDs
- Final citations

Generate the shared cases with:

```bash
PYTHONPATH=src python scripts/export_judge_cases.py \
  --validate-adapters
  ```
This command verifies that the exported cases can be converted into both RAGAS
SingleTurnSample objects and DeepEval LLMTestCase objects.

## RAGAS credential-free evaluation
The RAGAS integration evaluates retrieval using passage identifiers rather
than an LLM judge. This keeps the baseline reproducible and avoids requiring
hosted-model credentials.
Run it with:
```Bash
RAGAS_DO_NOT_TRACK=true \
PYTHONPATH=src \
python scripts/run_ragas_id_evaluation.py
```
Baseline results:
Metric	Result
Evaluated questions	16
Mean ID-based context precision	0.3542
Mean ID-based context recall	1.0000


Perfect context recall means every verified supporting passage appeared in the
retrieved context. Lower context precision reflects the retrieval of additional
candidate passages. The controlled workflow applies evidence selection after
retrieval, so raw context precision and final citation precision measure
different stages of the system.
The versioned report is stored at:
``text
artifacts/baselines/ragas-id-v0.1.json
``
## DeepEval credential-free evaluation
The DeepEval integration uses LLMTestCase and custom BaseMetric
implementations. The metrics are deterministic and require no judge model or
API credentials.
Run it with:
```Bash
PYTHONPATH=src python scripts/run_deepeval_deterministic.py
```
Baseline results:

Metric	Result
Evaluated questions	16
Mean answer correctness	0.5659
Answer-correctness pass rate	0.7500
Mean faithfulness	1.0000
Faithfulness pass rate	1.0000


Answer correctness uses normalized lexical token F1 against the verified
answer. The pass threshold is 0.5.
Faithfulness measures the proportion of normalized answer terms supported by
the retrieved evidence. The pass threshold is 0.8.
The versioned report is stored at:
```text
artifacts/baselines/deepeval-deterministic-v0.1.json
```

## Interpreting the results
The lexical correctness score is a conservative proxy. It can assign a lower
score to a valid answer when the generated answer paraphrases the verified
answer or provides a shorter supported response. It should therefore be
interpreted alongside faithfulness, citation correctness, and manual review.
The deterministic results do not claim semantic equivalence to an LLM judge.
They provide a stable, credential-free baseline suitable for local development
and continuous integration.
A hosted judge model can be added later for semantic correctness and
groundedness experiments. Those results should be tracked separately with the
model name, prompt version, temperature, token usage, latency, and estimated
cost.

## Dependency compatibility
The optional judge dependencies are installed with:
```Bash
python -m pip install -e ".[judge]"
```

The project pins compatible versions of RAGAS, DeepEval, and
LangChain Community because judge-framework APIs and transitive imports can
change between releases. Dependency upgrades should rerun adapter validation,
the complete test suite, and both saved evaluation reports.