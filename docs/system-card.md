# System Card

## System name

Enterprise RAG Reliability & Evaluation Platform

## Version

Portfolio baseline version 0.1

## System purpose

This platform evaluates the reliability of enterprise retrieval-augmented
generation systems before they are considered for production use.

The evaluated application is an enterprise IT support and policy assistant.
The platform measures whether retrieval, evidence selection, answer generation,
citation behavior, refusal behavior, and human escalation meet defined quality
requirements.

The platform is an evaluation and reliability system. It is not a general
chatbot and does not grant access, approve requests, modify accounts, or perform
administrative actions.

## Intended users

The intended users are:

- LLM and RAG engineers
- Machine-learning engineers
- AI platform teams
- Evaluation and quality engineers
- Security and responsible-AI reviewers
- Enterprise IT support stakeholders
- Technical recruiters reviewing the portfolio implementation

## Intended uses

The platform is designed to:

- Compare retrieval configurations
- Compare chunking configurations
- Measure evidence coverage and retrieval rank
- Evaluate answer correctness and faithfulness
- Detect unsupported claims and citation errors
- Measure refusal and escalation correctness
- Compare local or optional hosted models
- Track quality, latency, tokens, cost, and failures
- Detect regressions in continuous integration
- Support production-readiness reviews
- Preserve reproducible saved evaluation results

## Out-of-scope uses

The system should not be used to:

- Replace an enterprise authorization process
- Grant privileged access
- Perform account, device, or infrastructure changes
- Provide emergency or safety-critical instructions without human review
- Treat public GitLab policy as policy for another organization
- Make employment, legal, financial, medical, or disciplinary decisions
- Establish production readiness from the current starter benchmark alone
- Infer that a high aggregate score guarantees correctness on every question

## Evaluated application domain

The current benchmark covers public enterprise IT and security topics,
including:

- Password and account guidance
- Access requests
- Lost or stolen devices
- VPN guidance
- Device enrollment
- Phishing and suspicious messages
- Data classification
- AI support-tool restrictions
- Incident response
- Authorization-sensitive requests
- Out-of-domain and unsupported questions

The source material comes from public GitLab handbook pages. GitLab policy is
kept within its organizational context and is not presented as general policy
for other employers.

## Data and provenance

The evaluation data consists of:

- 10 public source records
- 19 independently collected corpus passages
- 20 manually verified benchmark questions
- 16 answerable generation cases
- Refusal and escalation cases
- Direct, ambiguous, adversarial, multi-document, unanswerable, and
  out-of-domain questions

Each source record includes:

- Source URL
- Publisher
- License
- Checked date
- Registered sections

Each answerable benchmark item includes:

- Verified answer
- Supporting source
- Supporting passage
- Source and section identifiers
- Category and difficulty
- Expected behavior

Synthetic policy documents and model-generated ground truth are not used.

The corpus remains separate from benchmark labels to reduce evaluation leakage.

## System components

The implemented system includes:

- Versioned document loading
- Metadata validation
- Stable content hashing
- Heading-aware, fixed-size, and recursive chunking
- BM25 retrieval
- Dense retrieval with Sentence Transformers and FAISS
- Hybrid reciprocal-rank fusion
- Cross-encoder reranking
- Heading-aware evidence selection
- Controlled LangGraph decisions
- Local Ollama model comparison
- RAGAS and DeepEval adapters
- MLflow experiment tracking
- OpenTelemetry tracing
- Structured JSON logs
- Streamlit reliability dashboard
- GitHub Actions evaluation gates
- Multi-stage Docker deployment

## Decision behavior

The controlled workflow returns one of three decisions.

### Answer

The system answers when:

- The question is in domain
- Retrieved evidence is sufficient
- The generated or extracted response is supported
- Citations refer to supplied evidence

### Refuse

The system refuses when:

- The question is outside the supported domain
- The requested factual claim is unsupported
- The question asks the assistant to bypass a documented restriction
- The system cannot responsibly provide the requested response

### Escalate

The system escalates when:

- The request requires human authorization
- A responsible team must make the decision
- The request involves unsupported administrative action
- The configured workflow identifies a human-review requirement

The assistant does not treat retrieved text or model output as authorization.

## Recorded performance

### Retrieval

| Configuration | Recall@3 | MRR |
|---|---:|---:|
| BM25 | 81.25% | 0.6875 |
| Dense | 93.75% | 0.8646 |
| Hybrid RRF | 93.75% | 0.8750 |
| Hybrid with reranking | 100% | 0.9688 |

### Chunking

Fixed 80-character chunks with 20-character overlap improved BM25 Recall@3
from 81.25% to 87.50% and MRR from 0.6875 to 0.6979. The configuration increased
the index from 19 to 37 chunks.

### Controlled workflow

The current deterministic workflow achieved 20 correct decisions across the
20-question starter benchmark.

### Answer quality

| Metric | Result |
|---|---:|
| Decision accuracy | 100% |
| Lexical answer correctness | 56.59% |
| Deterministic faithfulness | 100% |
| Hallucination rate | 0% |
| Citation precision | 100% |
| Citation recall | 100% |
| Refusal accuracy | 100% |
| Escalation accuracy | 100% |

### RAGAS and DeepEval

| Framework metric | Result |
|---|---:|
| RAGAS ID context precision | 35.42% |
| RAGAS ID context recall | 100% |
| DeepEval lexical correctness | 56.59% |
| DeepEval correctness pass rate | 75% |
| DeepEval deterministic faithfulness | 100% |

### Local model comparison

| Model | Correctness | Faithfulness | Citation precision | Citation recall | Mean latency |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 3B | 35.82% | 75.00% | 70.83% | 93.75% | 3.96 s |
| Mistral 7B | 56.18% | 76.53% | 93.75% | 100% | 7.23 s |
| Phi-4 Mini | 50.84% | 72.25% | 87.50% | 100% | 6.54 s |

All three local models achieved 100% structured-output reliability under the
tested constrained JSON schema.

## Metric interpretation

The recorded results apply only to the current data, configuration, machine,
and saved experiment runs.

Lexical correctness measures normalized token overlap. It can penalize valid
paraphrases and concise answers.

Deterministic faithfulness measures answer-term coverage by retrieved evidence.
It can miss semantic support expressed with different wording.

A zero measured hallucination rate does not prove that the system cannot
hallucinate. It means the deterministic proxy found no unsupported answer terms
in the recorded workflow baseline.

Aggregate metrics must be reviewed with individual failure cases.

## Safety and responsible-AI controls

The platform includes:

- Explicit answer, refusal, and escalation labels
- Human authorization boundaries
- Out-of-domain detection
- Evidence-quality checks
- Citation validation
- Unknown source-ID rejection
- Bounded workflow transitions
- No autonomous administrative actions
- Saved per-question failures
- CI regression gates
- Production-readiness warnings
- Optional hosted-model credentials
- Credential-free evaluation paths

## Human oversight

Human review is required for:

- Access and authorization requests
- Conflicting policy sources
- Missing or weak evidence
- Security-sensitive actions
- Production-readiness approval
- Benchmark creation and verification
- Source updates
- Threshold changes
- Material prompt or model changes

The system can recommend escalation but cannot complete the authorized action.

## Security and privacy

The public portfolio dataset does not contain private company documents,
customer records, real credentials, or personal support tickets.

The repository excludes:

- Environment files
- Hosted-model credentials
- Local MLflow databases
- OpenTelemetry trace outputs
- Structured application logs
- Virtual environments
- Local model weights
- Generated run artifacts not selected as versioned baselines

Docker images run as an unprivileged user.

A real enterprise deployment would additionally require:

- Authentication and role-based access control
- Secret management
- Encryption in transit and at rest
- Data retention and deletion policies
- Audit logging
- Tenant isolation
- Prompt-injection defenses
- Source-level access controls
- Private-data redaction
- Security review and threat modeling

## Known limitations

The current system has the following limitations:

- The benchmark contains only 20 questions.
- The source corpus represents one organization.
- The benchmark is a starter set rather than a held-out production sample.
- Some metrics are lexical deterministic proxies.
- Local latency was measured on one Mac.
- Load, concurrency, and sustained-throughput tests are not included.
- Hosted models have not yet been compared in the versioned baseline.
- Source conflict resolution is not comprehensively evaluated.
- Public source pages can change after their checked dates.
- Saved results can become stale if source content changes.
- The local pipeline image is large because it includes the full ML stack.

## Monitoring and observability

The platform records:

- Request and operation identifiers
- Retrieval configuration
- Retrieved documents
- Quality metrics
- Latency
- Token usage
- Estimated cost
- Structured-output validity
- Errors and failures
- Trace and span identifiers
- Escalation decisions

Production monitoring should additionally track:

- Traffic distribution
- Per-category failure rates
- Drift in question types
- Source freshness
- Refusal and escalation rates
- User-reported incidents
- Tail latency
- Provider availability
- Cost budgets
- Security events

## Change management

Changes to the following should trigger evaluation:

- Prompt text or version
- Model or provider
- Chunking strategy
- Embedding model
- Retrieval logic
- Reranker
- Evidence thresholds
- Decision rules
- Source content
- Benchmark labels

Pull requests run deterministic validation, tests, and quality gates. Material
changes should update versioned artifacts and document any score differences.

## Production-readiness conclusion

Current status: **NOT READY FOR PRODUCTION**

The platform demonstrates the engineering controls needed to evaluate a RAG
system, but the starter benchmark is too small to support a production
approval.

Before production consideration, the project requires:

- A larger held-out benchmark
- Independent human review
- Organization-specific private sources and access controls
- Semantic evaluation with judge reliability analysis
- Load and failure testing
- Security and privacy review
- Defined service-level objectives
- Operational ownership and incident procedures