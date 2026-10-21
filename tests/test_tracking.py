from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from rag_reliability import tracking


def sample_report() -> dict[str, Any]:
    return {
        "configuration": {
            "retriever": "hybrid_reranked",
            "top_k": 3,
            "candidate_k": 10,
            "metadata": {"ignored": "nested"},
        },
        "summary": {
            "evaluated_questions": 16,
            "recall_at_3": 1.0,
            "mean_reciprocal_rank": 0.96875,
        },
        "timing": {
            "total_query_time_ms": 320.0,
            "mean_query_time_ms": 20.0,
        },
        "questions": [],
    }


def test_scalar_parameters_excludes_nested_values() -> None:
    parameters = tracking.scalar_parameters(
        {
            "retriever": "hybrid",
            "top_k": 3,
            "enabled": True,
            "nested": {"candidate_k": 10},
            "items": ["dense", "bm25"],
        }
    )

    assert parameters == {
        "retriever": "hybrid",
        "top_k": 3,
        "enabled": True,
    }


def test_log_retrieval_report_records_run_data(
    monkeypatch,
) -> None:
    recorded: dict[str, Any] = {}

    @contextmanager
    def fake_start_run(*, run_name: str):
        recorded["run_name"] = run_name
        yield SimpleNamespace(
            info=SimpleNamespace(run_id="run-123")
        )

    monkeypatch.setattr(tracking.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        tracking.mlflow,
        "log_params",
        lambda values: recorded.update(params=values),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_metrics",
        lambda values: recorded.update(metrics=values),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tags",
        lambda values: recorded.update(tags=values),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_dict",
        lambda value, path: recorded.update(
            artifact=value,
            artifact_path=path,
        ),
    )

    report = sample_report()
    run_id = tracking.log_retrieval_report(
        "hybrid_reranked",
        report,
    )

    assert run_id == "run-123"
    assert recorded["run_name"] == "hybrid_reranked"
    assert recorded["params"]["top_k"] == 3
    assert "metadata" not in recorded["params"]
    assert recorded["metrics"]["recall_at_3"] == 1.0
    assert (
        recorded["metrics"]["mean_reciprocal_rank"]
        == 0.96875
    )
    assert recorded["tags"]["evaluation_type"] == "retrieval"
    assert (
        recorded["artifact_path"]
        == "reports/hybrid_reranked.json"
    )
    assert recorded["artifact"] == report