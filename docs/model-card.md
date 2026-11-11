# Model Card

## Evaluation name

Local Open-Source RAG Model Comparison

## Version

Baseline version 0.1

## Evaluated models

| Display name | Ollama identifier | Approximate category | Deployment |
|---|---|---|---|
| Llama 3.2 3B | `llama3.2:3b-instruct-q4_K_M` | Small local instruct model | Ollama |
| Mistral 7B | `mistral:7b` | Medium local instruct model | Ollama |
| Phi-4 Mini | `phi4-mini:latest` | Small local reasoning model | Ollama |

The repository does not redistribute model weights. Models are installed and
managed separately through Ollama.

Users are responsible for reviewing the upstream model licenses, acceptable-use
terms, and deployment requirements before using a model outside this portfolio
evaluation.

## Model task

The evaluated task is evidence-grounded answer generation for enterprise IT and
security questions.

Each model receives:

- One question
- A bounded list of retrieved evidence passages
- Exact source identifiers
- Instructions to use only supplied evidence
- A constrained JSON output schema

Each model must return:

```json
{
  "answer": "Evidence-grounded response",
  "citations": ["exact-source-id"]
}
```

# Model Card

## Evaluation name

Local Open-Source RAG Model Comparison

## Version

Baseline version 0.1

## Evaluated models

| Display name | Ollama identifier | Approximate category | Deployment |
|---|---|---|---|
| Llama 3.2 3B | `llama3.2:3b-instruct-q4_K_M` | Small local instruct model | Ollama |
| Mistral 7B | `mistral:7b` | Medium local instruct model | Ollama |
| Phi-4 Mini | `phi4-mini:latest` | Small local reasoning model | Ollama |

The repository does not redistribute model weights. Models are installed and
managed separately through Ollama.

Users are responsible for reviewing the upstream model licenses, acceptable-use
terms, and deployment requirements before using a model outside this portfolio
evaluation.

## Model task

The evaluated task is evidence-grounded answer generation for enterprise IT and
security questions.

Each model receives:

- One question
- A bounded list of retrieved evidence passages
- Exact source identifiers
- Instructions to use only supplied evidence
- A constrained JSON output schema

Each model must return:

```json
{
  "answer": "Evidence-grounded response",
  "citations": ["exact-source-id"]
}
```
The models are not asked to retrieve documents, classify authorization, or
decide whether a user should receive access. Those responsibilities remain in
the controlled application workflow.

## Intended use
The evaluated models are intended for:
- Local RAG prototyping
- Credential-free model comparison
- Evidence-grounded answer generation
- Structured-output reliability testing
- Citation-selection experiments
- Quality and latency tradeoff analysis
- Reduced demonstration mode when hosted APIs are unavailable
## Out-of-scope use
The models should not be used to:
- Grant access or authorize actions
- Execute IT support operations
- Make security or incident-management decisions without human review
- Answer without retrieved evidence
- Replace organization-specific policy review
- Process sensitive enterprise data without approved controls
- Provide production service based only on the current benchmark
- Treat generated citations as valid without application-level verification
## Evaluation dataset
The comparison uses 16 answerable cases from the verified enterprise benchmark.

The cases are derived from public GitLab handbook sources and contain:
- Verified questions
- Verified answers
- Retrieved evidence
- Reference evidence
- Exact passage identifiers
- Expected citations
Refusal and escalation cases are excluded from this generation comparison
because they are evaluated separately by the controlled LangGraph workflow.

No synthetic policy documents or model-generated ground truth are used.
## Evaluation configuration
The models were evaluated under the same controls:
- Provider: local Ollama
- Temperature: 0
- Seed: 42
- Same 16 questions
- Same retrieved evidence
- Same evidence order
- Same prompt instructions
- Same JSON schema
- Same deterministic evaluation functions
- One unrecorded warm-up request per model
- 16 recorded requests per model
The citation schema restricts outputs to source identifiers supplied in the
request. The application also rejects unknown source identifiers after
generation.
## Recorded results
Model	Structured output	Correctness	Faithfulness	Citation precision	Citation recall	Mean latency
Llama 3.2 3B	100%	35.82%	75.00%	70.83%	93.75%	3.96 s
Mistral 7B	100%	56.18%	76.53%	93.75%	100%	7.23 s
Phi-4 Mini	100%	50.84%	72.25%	87.50%	100%	6.54 s


## Token usage and estimated cost
Model	Input tokens	Output tokens	Estimated API cost
Llama 3.2 3B	4,261	677	$0.00
Mistral 7B	4,862	862	$0.00
Phi-4 Mini	3,900	818	$0.00


The API cost is recorded as zero because the models ran locally. Local inference still consumes memory, CPU or GPU resources, electricity, and engineering
time.
## Recommended model
# Quality-focused selection
Mistral 7B is the recommended local model for the current benchmark.

It achieved:
- Highest lexical answer correctness
- Highest deterministic faithfulness
- Highest citation precision
- Complete citation recall
- Zero structured-output failures
- Zero transport failures
The tradeoff is the highest mean latency among the tested models.
## Latency-focused selection
Llama 3.2 3B is the fastest tested model.

It may be useful for:
- Low-cost local development
- Smoke tests
- Interactive demonstrations
- Latency-sensitive experiments
Its lower correctness and citation precision make it less suitable for the
current quality-focused configuration.
## Balanced alternative
Phi-4 Mini provides a middle option between Llama and Mistral.

It achieved complete citation recall and lower latency than Mistral, but its
correctness, faithfulness, and citation precision remained below Mistral.
Structured-output reliability
An initial smoke test showed that Llama shortened a source identifier and Phi
cited unnecessary evidence.

The final configuration added:
- A JSON schema enum containing the allowed source IDs
- Unique citation values
- Explicit instructions to copy identifiers exactly
- Application-level unknown-ID validation
After these controls, all three models achieved 100% structured-output
reliability across the 16 recorded cases.

This result demonstrates the combined reliability of model behavior and
constrained decoding. It should not be interpreted as evidence that the models
will always produce valid JSON without those controls.
## Failure analysis
No model produced a transport failure or invalid structured output in the full recorded comparison.

Content-quality weaknesses remained.

Question q005 was one of the lowest-scoring correctness cases for every
model. This shared failure suggests a possible issue involving prompt wording,
retrieved context, answer conciseness, or the lexical metric.

Llama received zero lexical correctness on q014 and q018 while retaining
perfect deterministic faithfulness. These results may reflect supported
paraphrases that differ from the verified answer.

Phi-4 Mini received zero deterministic faithfulness on q004. That case
requires manual semantic review because lexical evidence coverage can miss
valid paraphrases.
## Metric definitions
# Answer correctness
Normalized lexical token F1 between the generated answer and verified answer.

This metric rewards shared meaningful terms but can penalize valid paraphrases, different levels of detail, and concise answers.
## Faithfulness
The proportion of normalized answer terms present in retrieved evidence.

This is a deterministic evidence-coverage proxy. It does not fully measure
semantic entailment.
## Citation precision
The proportion of generated citations that match verified supporting passage
identifiers Citation recall
The proportion of verified supporting passage identifiers included in the
generated citations.
## Structured-output reliability
The proportion of responses that:
- Parse as JSON
- Contain exactly answer and citations
- Include a non-empty string answer
- Include a string-array citation field
- Use only supplied source identifiers
## Latency interpretation
Latency was measured on one local Mac after an unrecorded warm-up request.

The measurements do not represent:
- Production server latency
- Concurrent throughput
- Tail latency
- Cold-start performance
- GPU-serving performance
- Hosted-provider latency
- Network overhead
The results are useful for comparison within the recorded local environment.
## Model limitations
The evaluated models may:
- Produce unsupported paraphrases
- Cite excessive evidence
- Omit required evidence
- Copy source text without sufficient synthesis
- Misinterpret negation or conditions
- Generate plausible but incorrect answers
- Behave differently after model updates
- Behave differently with longer contexts
- Show different performance on private enterprise documents
- Degrade under concurrent load
Constrained output prevents some formatting and citation-ID errors, but it does
not guarantee factual correctness.
## Bias and fairness considerations
The benchmark evaluates technical policy questions rather than demographic
decisions. It does not provide meaningful evidence about fairness across
languages, dialects, accessibility needs, cultures, or demographic groups.

The source corpus reflects one organization's terminology and processes. Model performance may be biased toward wording that resembles that corpus.

A production evaluation should include:
- Diverse user phrasing
- Non-native English questions
- Accessibility-oriented queries
- Ambiguous abbreviations
- Organization-specific vocabulary
- Adversarial and prompt-injection cases
- Questions from different user roles
## Privacy and security
The recorded benchmark contains public information and no real customer data,
credentials, private tickets, or employee records.

Production use would require controls for:
- Sensitive-data redaction
- Prompt and response retention
- Model-host trust
- Source-level access permissions
- Tenant isolation
- Audit logging
- Secret management
- Incident response
- Model and dependency vulnerability review
## Reproducibility
The versioned report is stored at:
```text
artifacts/baselines/model-comparison-local-v0.1.json
```
Run the comparison with:
```bash
caffeinate -i env PYTHONPATH=src \
  python scripts/run_model_comparison.py
```
Required Ollama models must already be installed.

The model tag latest can change over time. Future experiment releases should
record immutable model digests in addition to human-readable model names.
## Model-selection conclusion
Current local recommendation: Mistral 7B for quality-focused evaluation

Current fast local option: Llama 3.2 3B

Current balanced alternative: Phi-4 Mini

These recommendations apply only to the recorded 16-case starter benchmark.
They are experiment findings, not general rankings of the underlying models.