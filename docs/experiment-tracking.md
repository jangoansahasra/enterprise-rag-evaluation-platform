# Experiment tracking and observability

The retrieval evaluation pipeline records experiment metadata in MLflow and emits structured application logs and OpenTelemetry traces.

## MLflow tracking

Each retrieval configuration creates a separate MLflow run containing:

- Retriever and model configuration
- Candidate and final result counts
- Recall@3
- Mean reciprocal rank
- Total evaluation latency
- Mean query latency
- Complete JSON evaluation report
- Project, evaluation-type, and data-source tags

The local tracking database and run artifacts are stored under `artifacts/mlflow/`. This directory is excluded from Git because MLflow runs are generated artifacts.

Run the retrieval experiment:

~~~bash
PYTHONPATH=src python scripts/run_retrieval_experiments.py
~~~

Open the local MLflow interface:

~~~bash
python -m mlflow ui \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --default-artifact-root ./artifacts/mlflow/artifacts \
  --port 5000
~~~

Then visit `http://127.0.0.1:5000`.

## Structured logs

Application events are written as JSON Lines to:

~~~text
artifacts/observability/application.jsonl
~~~

Each evaluated retrieval configuration records:

- Request ID
- Operation name
- Retriever name
- Benchmark size
- Retrieval depth
- Recall@3
- Mean reciprocal rank
- Mean query latency
- Completion or failure status

## OpenTelemetry traces

Completed spans are written to:

~~~text
artifacts/observability/traces.jsonl
~~~

Each span includes its trace ID, span ID, request ID, operation latency, retriever configuration, and status. The shared request ID connects application events to the corresponding trace.

The local JSON exporter keeps the project credential-free while preserving standard OpenTelemetry instrumentation that can later be connected to systems such as Jaeger, Grafana Tempo, or an OTLP-compatible observability platform.

## Reproducibility

MLflow and observability outputs are intentionally excluded from version control. Versioned benchmark and baseline JSON files remain in the repository for immediate review, while local runs can be reproduced from the documented commands.