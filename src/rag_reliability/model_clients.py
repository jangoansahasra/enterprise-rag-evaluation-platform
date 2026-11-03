"""Provider-neutral model clients for controlled RAG comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable
from urllib.request import Request, urlopen


JsonTransport = Callable[..., Any]


@dataclass(frozen=True)
class ModelResult:
    """Normalized result returned by every model provider."""

    provider: str
    model: str
    answer: str
    citations: tuple[str, ...]
    structured_output_valid: bool
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float
    raw_output: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OllamaClient:
    """Generate grounded structured answers through local Ollama."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        transport: JsonTransport = urlopen,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def generate(
        self,
        question: str,
        contexts: list[dict[str, str]],
    ) -> ModelResult:
        """Generate one answer using only the supplied evidence."""
        prompt = self._build_prompt(question, contexts)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
    "type": "array",
    "items": {
        "type": "string",
        "enum": [
            context["id"] for context in contexts
        ],
    },
    "uniqueItems": True,
},
                },
                "required": ["answer", "citations"],
                "additionalProperties": False,
            },
            "options": {
                "temperature": 0,
                "seed": 42,
            },
        }

        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = perf_counter()

        try:
            with self.transport(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                provider_response = json.loads(
                    response.read().decode("utf-8")
                )

            latency_ms = (perf_counter() - started) * 1000
            raw_output = str(provider_response.get("response", ""))
            parsed, valid, parsing_error = self._parse_output(
                raw_output,
                contexts,
            )

            return ModelResult(
                provider=self.provider,
                model=self.model,
                answer=parsed.get("answer", ""),
                citations=tuple(parsed.get("citations", [])),
                structured_output_valid=valid,
                latency_ms=latency_ms,
                input_tokens=self._optional_int(
                    provider_response.get("prompt_eval_count")
                ),
                output_tokens=self._optional_int(
                    provider_response.get("eval_count")
                ),
                estimated_cost_usd=0.0,
                raw_output=raw_output,
                error=parsing_error,
            )
        except Exception as error:
            latency_ms = (perf_counter() - started) * 1000
            return ModelResult(
                provider=self.provider,
                model=self.model,
                answer="",
                citations=(),
                structured_output_valid=False,
                latency_ms=latency_ms,
                input_tokens=None,
                output_tokens=None,
                estimated_cost_usd=0.0,
                raw_output="",
                error=f"{type(error).__name__}: {error}",
            )

    @staticmethod
    def _build_prompt(
        question: str,
        contexts: list[dict[str, str]],
    ) -> str:
        evidence_blocks = "\n\n".join(
            (
                f"SOURCE_ID: {context['id']}\n"
                f"CONTENT: {context['text']}"
            )
            for context in contexts
        )

        return (
            "You are evaluating an enterprise IT support RAG system.\n"
            "Answer the question using only the supplied evidence.\n"
            "Do not add facts that are absent from the evidence.\n"
            "Use only sources that directly support the answer.\n"
"Copy each cited SOURCE_ID exactly, character for character.\n"
"Never shorten, rewrite, or invent a SOURCE_ID.\n"
"Do not cite irrelevant evidence.\n"
"Return JSON with exactly two fields: answer and citations.\n"
"The citations field must be a JSON array containing only "
"the exact SOURCE_ID values provided below.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n{evidence_blocks}"
        )

    @staticmethod
    def _parse_output(
        raw_output: str,
        contexts: list[dict[str, str]],
    ) -> tuple[dict[str, Any], bool, str | None]:
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as error:
            return {}, False, f"Invalid JSON: {error}"

        if not isinstance(parsed, dict):
            return {}, False, "Structured output must be a JSON object."

        if set(parsed) != {"answer", "citations"}:
            return (
                parsed,
                False,
                "Structured output must contain only answer and citations.",
            )

        answer = parsed.get("answer")
        citations = parsed.get("citations")

        if not isinstance(answer, str) or not answer.strip():
            return parsed, False, "Answer must be a non-empty string."

        if (
            not isinstance(citations, list)
            or not all(
                isinstance(citation, str)
                for citation in citations
            )
        ):
            return (
                parsed,
                False,
                "Citations must be an array of strings.",
            )

        allowed_ids = {
            context["id"] for context in contexts
        }
        unknown_citations = sorted(
            set(citations) - allowed_ids
        )

        if unknown_citations:
            return (
                parsed,
                False,
                "Unknown citation IDs: "
                + ", ".join(unknown_citations),
            )

        return parsed, True, None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None