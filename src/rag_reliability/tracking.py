"""Local MLflow tracking for reproducible RAG experiments."""

from __future__ import annotations
from mlflow import MlflowClient
from pathlib import Path
from typing import Any

import mlflow


DEFAULT_EXPERIMENT_NAME = "enterprise-rag-retrieval"


def configure_mlflow(
    tracking_directory: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> None:
    """Configure a credential-free local MLflow SQLite store."""
    tracking_directory.mkdir(parents=True, exist_ok=True)

    database_path = (tracking_directory / "mlflow.db").resolve()
    artifact_path = (tracking_directory / "artifacts").resolve()

    mlflow.set_tracking_uri(f"sqlite:///{database_path}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_path.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)


def scalar_parameters(configuration: dict[str, Any]) -> dict[str, Any]:
    """Return configuration values that MLflow can store as parameters."""
    return {
        key: value
        for key, value in configuration.items()
        if isinstance(value, (str, int, float, bool))
    }


def log_retrieval_report(
    run_name: str,
    report: dict[str, Any],
) -> str:
    """Record one retrieval configuration and return its MLflow run ID."""
    configuration = report["configuration"]
    summary = report["summary"]
    timing = report["timing"]

    parameters = scalar_parameters(configuration)
    parameters["configuration_name"] = run_name

    metrics = {
        "evaluated_questions": float(summary["evaluated_questions"]),
        "recall_at_3": float(summary["recall_at_3"]),
        "mean_reciprocal_rank": float(
            summary["mean_reciprocal_rank"]
        ),
        "total_query_time_ms": float(timing["total_query_time_ms"]),
        "mean_query_time_ms": float(timing["mean_query_time_ms"]),
    }

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "enterprise-rag-reliability",
                "evaluation_type": "retrieval",
                "data_source": "public-gitlab-handbook",
            }
        )
        mlflow.log_dict(report, f"reports/{run_name}.json")
        return active_run.info.run_id


def log_retrieval_experiment(
    reports: dict[str, dict[str, Any]],
    tracking_directory: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, str]:
    """Record all retrieval configurations as separate MLflow runs."""
    configure_mlflow(tracking_directory, experiment_name)

    return {
        name: log_retrieval_report(name, report)
        for name, report in reports.items()
    }