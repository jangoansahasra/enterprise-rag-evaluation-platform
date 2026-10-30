"""Run credential-free deterministic metrics through DeepEval interfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from rag_reliability.answer_evaluation import (
    evidence_support,
    token_f1,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "artifacts/runs/judge-cases.json"
OUTPUT_PATH = (
    ROOT / "artifacts/runs/deepeval-deterministic.json"
)


class LexicalAnswerCorrectnessMetric(BaseMetric):
    """Measure normalized lexical agreement with the verified answer."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-token-f1"

    def measure(
        self,
        test_case: LLMTestCase,
        *args: object,
        **kwargs: object,
    ) -> float:
        actual_output = test_case.actual_output or ""
        expected_output = test_case.expected_output or ""

        self.score = token_f1(
            actual_output,
            expected_output,
        )
        self.success = self.score >= self.threshold
        self.reason = (
            "Normalized lexical token F1 between the generated "
            "answer and the verified answer."
        )
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: object,
        **kwargs: object,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Lexical Answer Correctness"


class EvidenceFaithfulnessMetric(BaseMetric):
    """Measure generated-answer coverage by retrieved evidence."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-evidence-support"

    def measure(
        self,
        test_case: LLMTestCase,
        *args: object,
        **kwargs: object,
    ) -> float:
        actual_output = test_case.actual_output or ""
        retrieval_context = [
            str(context)
            for context in (test_case.retrieval_context or [])
        ]

        self.score = evidence_support(
            actual_output,
            retrieval_context,
        )
        self.success = self.score >= self.threshold
        self.reason = (
            "Fraction of normalized answer terms supported by "
            "the retrieved evidence."
        )
        return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        *args: object,
        **kwargs: object,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Evidence Faithfulness"


def evaluate_case(case: dict[str, object]) -> dict[str, object]:
    """Evaluate one exported judge case with deterministic metrics."""
    test_case = LLMTestCase(
        input=str(case["question"]),
        actual_output=str(case["actual_output"]),
        expected_output=str(case["expected_output"]),
        retrieval_context=[
            str(context)
            for context in case["retrieval_context"]
        ],
        metadata={
            "question_id": str(case["question_id"]),
        },
    )

    correctness = LexicalAnswerCorrectnessMetric()
    faithfulness = EvidenceFaithfulnessMetric()

    correctness_score = correctness.measure(test_case)
    faithfulness_score = faithfulness.measure(test_case)

    return {
        "question_id": case["question_id"],
        "question": case["question"],
        "answer_correctness": correctness_score,
        "answer_correctness_threshold": correctness.threshold,
        "answer_correctness_passed": correctness.is_successful(),
        "faithfulness": faithfulness_score,
        "faithfulness_threshold": faithfulness.threshold,
        "faithfulness_passed": faithfulness.is_successful(),
    }


def main() -> None:
    """Load judge cases, run DeepEval metrics, and save the report."""
    if not INPUT_PATH.exists():
        raise SystemExit(
            "Judge-case input is missing. Run "
            "`PYTHONPATH=src python scripts/export_judge_cases.py` "
            "first."
        )

    source_report = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )
    evaluated_cases = [
        evaluate_case(case)
        for case in source_report["cases"]
    ]

    correctness_scores = [
        float(case["answer_correctness"])
        for case in evaluated_cases
    ]
    faithfulness_scores = [
        float(case["faithfulness"])
        for case in evaluated_cases
    ]

    summary = {
        "evaluated_questions": len(evaluated_cases),
        "mean_answer_correctness": mean(correctness_scores),
        "answer_correctness_pass_rate": mean(
            float(case["answer_correctness_passed"])
            for case in evaluated_cases
        ),
        "mean_faithfulness": mean(faithfulness_scores),
        "faithfulness_pass_rate": mean(
            float(case["faithfulness_passed"])
            for case in evaluated_cases
        ),
    }

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "framework": "DeepEval",
        "evaluation_type": "credential_free_deterministic",
        "requires_judge_model": False,
        "source_cases": "artifacts/runs/judge-cases.json",
        "metrics": {
            "answer_correctness": {
                "implementation": "normalized lexical token F1",
                "threshold": 0.5,
            },
            "faithfulness": {
                "implementation": "answer-term evidence coverage",
                "threshold": 0.8,
            },
        },
        "summary": summary,
        "cases": evaluated_cases,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("DeepEval credential-free evaluation")
    print("-" * 46)
    print(
        f"Evaluated questions:             "
        f"{summary['evaluated_questions']}"
    )
    print(
        f"Mean answer correctness:         "
        f"{summary['mean_answer_correctness']:.4f}"
    )
    print(
        f"Answer correctness pass rate:    "
        f"{summary['answer_correctness_pass_rate']:.4f}"
    )
    print(
        f"Mean faithfulness:               "
        f"{summary['mean_faithfulness']:.4f}"
    )
    print(
        f"Faithfulness pass rate:          "
        f"{summary['faithfulness_pass_rate']:.4f}"
    )
    print()
    print(f"Saved report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()