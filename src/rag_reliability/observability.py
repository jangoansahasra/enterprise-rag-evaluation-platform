"""Structured logging and local OpenTelemetry tracing."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


LOGGER_NAME = "rag_reliability"


class JsonFormatter(logging.Formatter):
    """Format application events as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        event_fields = getattr(record, "event_fields", {})
        payload.update(event_fields)
        return json.dumps(payload, default=str, sort_keys=True)


class JsonFileSpanExporter(SpanExporter):
    """Write completed OpenTelemetry spans to a JSON Lines file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        spans: list[ReadableSpan],
    ) -> SpanExportResult:
        with self.output_path.open("a", encoding="utf-8") as output:
            for span in spans:
                context = span.get_span_context()
                payload = {
                    "trace_id": format(context.trace_id, "032x"),
                    "span_id": format(context.span_id, "016x"),
                    "name": span.name,
                    "start_time_unix_nano": span.start_time,
                    "end_time_unix_nano": span.end_time,
                    "status": span.status.status_code.name,
                    "attributes": dict(span.attributes or {}),
                }
                output.write(
                    json.dumps(payload, default=str, sort_keys=True)
                    + "\n"
                )

        return SpanExportResult.SUCCESS


def configure_json_logging(output_path: Path) -> logging.Logger:
    """Configure the project logger with a JSON Lines file handler."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers:
        if getattr(handler, "_rag_json_handler", False):
            return logger

    handler = logging.FileHandler(output_path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler._rag_json_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def configure_tracing(output_path: Path) -> trace.Tracer:
    """Configure local OpenTelemetry tracing with a JSON exporter."""
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "enterprise-rag-reliability"}
        )
    )
    provider.add_span_processor(
        SimpleSpanProcessor(JsonFileSpanExporter(output_path))
    )
    return provider.get_tracer("rag_reliability")


def log_event(
    logger: logging.Logger,
    event: str,
    **fields: Any,
) -> None:
    """Write a named structured event."""
    logger.info(
        event,
        extra={
            "event_fields": {
                "event": event,
                **fields,
            }
        },
    )


@contextmanager
def observed_operation(
    tracer: trace.Tracer,
    logger: logging.Logger,
    operation_name: str,
    *,
    request_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Trace an operation and record its result and latency."""
    current_request_id = request_id or str(uuid4())
    span_attributes = {
        "request.id": current_request_id,
        **(attributes or {}),
    }

    start = perf_counter()
    log_event(
        logger,
        "operation_started",
        request_id=current_request_id,
        operation=operation_name,
        **(attributes or {}),
    )

    with tracer.start_as_current_span(
        operation_name,
        attributes=span_attributes,
    ) as span:
        try:
            yield current_request_id
        except Exception as error:
            span.record_exception(error)
            span.set_status(
                trace.Status(
                    trace.StatusCode.ERROR,
                    str(error),
                )
            )
            log_event(
                logger,
                "operation_failed",
                request_id=current_request_id,
                operation=operation_name,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            latency_ms = round((perf_counter() - start) * 1000, 3)
            span.set_attribute("operation.latency_ms", latency_ms)
            log_event(
                logger,
                "operation_completed",
                request_id=current_request_id,
                operation=operation_name,
                latency_ms=latency_ms,
            )


def configure_observability(
    output_directory: Path,
) -> tuple[logging.Logger, trace.Tracer]:
    """Configure local structured logs and traces."""
    output_directory.mkdir(parents=True, exist_ok=True)

    logger = configure_json_logging(
        output_directory / "application.jsonl"
    )
    tracer = configure_tracing(
        output_directory / "traces.jsonl"
    )
    return logger, tracer