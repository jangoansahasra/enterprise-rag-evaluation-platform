"""Tests for provider-neutral model clients."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request

from rag_reliability.model_clients import OllamaClient


class FakeResponse:
    """Context-managed response returned by the fake transport."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class RecordingTransport:
    """Return a configured response and retain the HTTP request."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.request: Request | None = None
        self.timeout: float | None = None

    def __call__(
        self,
        request: Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        return FakeResponse(self.payload)


CONTEXTS = [
    {
        "id": "policy::passwords",
        "text": "The approved password manager is 1Password.",
    },
    {
        "id": "policy::support",
        "text": "Contact IT support if access cannot be restored.",
    },
]


def test_ollama_client_returns_normalized_result() -> None:
    transport = RecordingTransport(
        {
            "response": json.dumps(
                {
                    "answer": "The approved manager is 1Password.",
                    "citations": ["policy::passwords"],
                }
            ),
            "prompt_eval_count": 125,
            "eval_count": 18,
        }
    )
    client = OllamaClient(
        "llama3.2:3b",
        transport=transport,
    )

    result = client.generate(
        "Which password manager is approved?",
        CONTEXTS,
    )

    assert result.provider == "ollama"
    assert result.model == "llama3.2:3b"
    assert result.answer == "The approved manager is 1Password."
    assert result.citations == ("policy::passwords",)
    assert result.structured_output_valid is True
    assert result.input_tokens == 125
    assert result.output_tokens == 18
    assert result.estimated_cost_usd == 0.0
    assert result.error is None
    assert result.latency_ms >= 0

    assert transport.request is not None
    assert transport.request.full_url.endswith("/api/generate")
    assert transport.timeout == 120.0

    request_payload = json.loads(
        transport.request.data.decode("utf-8")
    )
    assert request_payload["model"] == "llama3.2:3b"
    assert request_payload["stream"] is False
    assert request_payload["options"]["temperature"] == 0
    citation_schema = request_payload["format"]["properties"][
    "citations"
]
    assert citation_schema["uniqueItems"] is True
    assert citation_schema["items"]["enum"] == [
    "policy::passwords",
    "policy::support",
]
    assert "policy::passwords" in request_payload["prompt"]


def test_unknown_citation_makes_output_invalid() -> None:
    transport = RecordingTransport(
        {
            "response": json.dumps(
                {
                    "answer": "Use 1Password.",
                    "citations": ["invented::source"],
                }
            ),
            "prompt_eval_count": 50,
            "eval_count": 10,
        }
    )
    client = OllamaClient(
        "phi4-mini:latest",
        transport=transport,
    )

    result = client.generate(
        "Which password manager is approved?",
        CONTEXTS,
    )

    assert result.structured_output_valid is False
    assert result.citations == ("invented::source",)
    assert result.error == "Unknown citation IDs: invented::source"


def test_malformed_json_is_recorded_as_failure() -> None:
    transport = RecordingTransport(
        {
            "response": "This is not JSON.",
            "prompt_eval_count": 30,
            "eval_count": 7,
        }
    )
    client = OllamaClient(
        "mistral:7b",
        transport=transport,
    )

    result = client.generate(
        "Which password manager is approved?",
        CONTEXTS,
    )

    assert result.answer == ""
    assert result.citations == ()
    assert result.structured_output_valid is False
    assert result.error is not None
    assert result.error.startswith("Invalid JSON:")


def test_extra_output_fields_are_rejected() -> None:
    transport = RecordingTransport(
        {
            "response": json.dumps(
                {
                    "answer": "Use 1Password.",
                    "citations": ["policy::passwords"],
                    "confidence": 0.99,
                }
            ),
        }
    )
    client = OllamaClient(
        "llama3.2:3b",
        transport=transport,
    )

    result = client.generate(
        "Which password manager is approved?",
        CONTEXTS,
    )

    assert result.structured_output_valid is False
    assert result.error == (
        "Structured output must contain only answer and citations."
    )


def test_transport_error_is_returned_without_crashing() -> None:
    def failing_transport(
        request: Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        raise ConnectionError("Ollama is unavailable")

    client = OllamaClient(
        "llama3.2:3b",
        transport=failing_transport,
    )

    result = client.generate(
        "Which password manager is approved?",
        CONTEXTS,
    )

    assert result.answer == ""
    assert result.citations == ()
    assert result.structured_output_valid is False
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.error == (
        "ConnectionError: Ollama is unavailable"
    )