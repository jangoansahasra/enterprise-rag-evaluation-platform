# Local model comparison

This experiment compares three local open-source models on the same 16
verified, answerable enterprise RAG cases.

All models receive identical questions, retrieved evidence, prompts, decoding
settings, and structured-output schemas. The experiment runs through Ollama
without hosted-model credentials or API charges.

## Models

| Model | Parameters | Quantization | Deployment |
|---|---:|---|---|
| Llama 3.2 Instruct | 3.2B | Q4_K_M | Local Ollama |
| Mistral | 7B | Local Ollama model | Local Ollama |
| Phi-4 Mini | Mini model | Local Ollama model | Local Ollama |

The comparison uses temperature `0`, seed `42`, an identical evidence-grounded
prompt, and a constrained JSON schema containing an answer and verified source
identifiers.

## Run the experiment

Ollama must be installed and running with the configured models available.

```bash
ollama list
```
Run the complete comparison:

```bash
caffeinate -i env PYTHONPATH=src \
  python scripts/run_model_comparison.py
```
Run a smaller smoke test:

```bash
PYTHONPATH=src python scripts/run_model_comparison.py \
  --limit 1 \
  --skip-warmup \
  --output artifacts/runs/model-comparison-smoke.json
```
The runner saves progress after every case. A partially completed report remains available if the process is interrupted.

## Results

Model	Structured output	Answer correctness	Faithfulness	Citation precision	Citation recall	Mean latency
Llama 3.2 3B	100%	35.82%	75.00%	70.83%	93.75%	3.96 s
Mistral 7B	100%	56.18%	76.53%	93.75%	100%	7.23 s
Phi-4 Mini	100%	50.84%	72.25%	87.50%	100%	6.54 s


All three models produced valid structured output for all 16 cases after the
citation schema was constrained to the supplied source identifiers.

## Token usage and cost

Model	Input tokens	Output tokens	Estimated API cost
Llama 3.2 3B	4,261	677	$0.00
Mistral 7B	4,862	862	$0.00
Phi-4 Mini	3,900	818	$0.00


The reported cost is zero because inference ran locally. Local execution still uses compute, memory, and electricity, so zero API cost does not mean zero operational cost.

## Quality and latency tradeoff

Mistral 7B produced the strongest overall quality. It achieved the highest
answer-correctness score, highest faithfulness score, 93.75% citation
precision, and complete citation recall. Its mean latency was 7.23 seconds,
making it the slowest tested model.
Llama 3.2 3B was the fastest model at 3.96 seconds per request. Its lower answer
correctness and citation precision make it more suitable for latency-sensitive
experiments than for the current quality-focused configuration.
Phi-4 Mini provided a middle position. It was faster than Mistral and achieved
complete citation recall, but its correctness, faithfulness, and citation
precision remained below Mistral.
For the current benchmark, Mistral 7B offers the best quality and reliability
balance when a mean generation latency of approximately seven seconds is
acceptable.

## Failure analysis
All models completed every case without transport or structured-output
failures. The remaining weaknesses were related to answer content and evidence
selection.
Question q005 was among the lowest-scoring correctness cases for all three
models. This common failure suggests that the prompt, retrieved context, or
benchmark wording should be reviewed before attributing the problem to one
model.
Llama also received zero lexical correctness on q014 and q018. Because
faithfulness remained perfect for those cases, the low score may reflect
paraphrasing or answer-format differences rather than unsupported content.
Phi-4 Mini received zero deterministic faithfulness on q004. That case
requires manual review because the lexical evidence-support metric can miss
supported paraphrases.

## Metric limitations
Answer correctness uses normalized lexical token F1 against the verified
answer. It can penalize concise answers and valid paraphrases.
Faithfulness measures answer-term coverage by retrieved evidence. It is a
deterministic proxy and does not prove semantic groundedness.
Citation precision and recall compare generated source identifiers with
verified passage identifiers. They measure citation selection more directly
than lexical answer metrics.
Latency was measured on one local Mac after an unrecorded warm-up request.
These numbers are useful for comparison in this environment but are not
production throughput or load-test measurements.

## Reproducibility
The versioned result is stored at:
```text
artifacts/baselines/model-comparison-local-v0.1.json
```
Hosted-model comparisons can be added later through the same normalized result format. Those experiments should record provider, model version, prompt version, token pricing, estimated cost, rate-limit failures, and deployment region separately from this local baseline.