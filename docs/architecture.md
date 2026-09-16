# Architecture

The platform separates the system under test from the evaluation system so the same benchmark can compare retrieval, prompts, and models without changing the scoring contract.

```mermaid
flowchart LR
    S[Versioned public sources] --> P[Document processing]
    P --> I[BM25, dense, and hybrid indexes]
    Q[Gold benchmark] --> E[Experiment runner]
    I --> W[Controlled LangGraph workflow]
    W --> E
    E --> M[Deterministic and judge metrics]
    M --> A[Versioned result artifacts]
    M --> T[MLflow and traces]
    A --> D[Streamlit reliability dashboard]
    M --> G{Production gates}
    G -->|pass| R[Candidate configuration]
    G -->|fail| F[Failure analysis]
```

## Evaluation contract

Retrieval is evaluated independently from generation using source and section identifiers. Generation is evaluated against verified answers, evidence passages, citation identifiers, and expected `answer`, `refuse`, or `escalate` behavior. Safety-critical failures remain individual gates rather than being hidden inside an average score.

The default path is credential-free: BM25, saved experiment artifacts, and later a local embedding model and local generator. Hosted models are optional adapters used for comparison.

## Deliberate scope

The first corpus uses one organization's public handbook. This preserves policy coherence and makes conflicting-source behavior meaningful. Additional organizations or TechQA can be added later as separate datasets, never mixed into the same policy namespace.
