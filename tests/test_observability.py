import json
from pathlib import Path

from rag_reliability.observability import (
    configure_observability,
    observed_operation,
)


def test_observed_operation_writes_logs_and_trace(
    tmp_path: Path,
) -> None:
    logger, tracer = configure_observability(tmp_path)

    with observed_operation(
        tracer,
        logger,
        "test_operation",
        request_id="request-123",
        attributes={"retriever": "bm25"},
    ):
        pass

    log_records = [
        json.loads(line)
        for line in (tmp_path / "application.jsonl")
        .read_text()
        .splitlines()
    ]
    trace_records = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl")
        .read_text()
        .splitlines()
    ]

    assert [record["event"] for record in log_records] == [
        "operation_started",
        "operation_completed",
    ]
    assert log_records[0]["request_id"] == "request-123"
    assert trace_records[0]["name"] == "test_operation"
    assert trace_records[0]["attributes"]["request.id"] == "request-123"
    assert trace_records[0]["attributes"]["retriever"] == "bm25"