# Evidence-selection improvement

The answer-quality baseline identified two citation failures even though the workflow achieved 100% decision accuracy:

- `q015` selected a general suspicious-message procedure instead of the Gmail-specific PhishArm procedure.
- `q019` returned only one of two sources required for a multi-document policy answer.

## Heading-aware selection

A heading-aware selector now evaluates the top cross-encoder candidates using:

- Original cross-encoder rank
- Query-term coverage in the section heading
- Query-term coverage in the passage
- A minimum lexical advantage for promotion
- A maximum permitted cross-encoder score drop

The score-drop guard prevents a lexical candidate from replacing a result when the cross-encoder strongly prefers the original result. This preserved correct results for lost-device reporting and RED data classification while resolving the near-tie for the Gmail phishing procedure.

## Controlled multi-document evidence

For conditional policy questions, the LangGraph workflow can add one related evidence passage when it contributes question-relevant terms. The behavior remains bounded:

- At most one additional passage is selected.
- The passage must overlap with meaningful question terms.
- Authorization and out-of-domain routes remain unchanged.
- The workflow does not start an autonomous retrieval loop.

This allows the AI data-handling question to cite both the approved-tool policy and the credential-classification policy.

## Results

| Metric | Baseline v0.1 | Improved v0.2 |
|---|---:|---:|
| Decision accuracy | 1.0000 | 1.0000 |
| Answer correctness | 0.5618 | 0.5659 |
| Faithfulness | 1.0000 | 1.0000 |
| Hallucination rate | 0.0000 | 0.0000 |
| Citation precision | 0.9375 | 1.0000 |
| Citation recall | 0.9062 | 1.0000 |
| Refusal accuracy | 1.0000 | 1.0000 |
| Escalation accuracy | 1.0000 | 1.0000 |

The results are measured on the 20-question starter benchmark. They demonstrate regression-driven development but do not establish production readiness.

## Reproduce

~~~bash
PYTHONPATH=src python scripts/evaluate_answer_quality.py
~~~

The improved versioned report is stored at:

~~~text
artifacts/baselines/answer-quality-v0.2.json
~~~