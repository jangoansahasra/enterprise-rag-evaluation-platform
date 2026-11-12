"""Tests for directional CI quality-gate behavior."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_gate_module() -> ModuleType:
    """Load the gate script without executing its main function."""
    sys.path.insert(0, str(SCRIPTS))

    try:
        spec = importlib.util.spec_from_file_location(
            "check_quality_gates",
            SCRIPTS / "check_quality_gates.py",
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


GATES = load_gate_module()


def test_minimum_gate_accepts_equal_and_higher_values() -> None:
    assert GATES.check_passed(
        0.95,
        "minimum",
        0.95,
    )
    assert GATES.check_passed(
        1.0,
        "minimum",
        0.95,
    )
    assert not GATES.check_passed(
        0.94,
        "minimum",
        0.95,
    )


def test_maximum_gate_accepts_equal_and_lower_values() -> None:
    assert GATES.check_passed(
        0.05,
        "maximum",
        0.05,
    )
    assert GATES.check_passed(
        0.0,
        "maximum",
        0.05,
    )
    assert not GATES.check_passed(
        0.06,
        "maximum",
        0.05,
    )


def test_build_checks_includes_live_rag_metrics() -> None:
    observed = {
        "sources": 10.0,
        "corpus_passages": 19.0,
        "benchmark_questions": 20.0,
        "recall_at_3": 0.8125,
        "mean_reciprocal_rank": 0.6875,
        "decision_accuracy": 1.0,
        "answer_correctness": 0.5659,
        "faithfulness": 1.0,
        "hallucination_rate": 0.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "refusal_accuracy": 1.0,
        "escalation_accuracy": 1.0,
    }
    thresholds = {
        "dataset": {
            "minimum_sources": 10,
            "minimum_corpus_passages": 19,
            "minimum_benchmark_questions": 20,
        },
        "retrieval": {
            "minimum_recall_at_3": 0.8,
            "minimum_mean_reciprocal_rank": 0.65,
        },
        "workflow": {
            "minimum_decision_accuracy": 0.95,
        },
        "answer_quality": {
            "minimum_answer_correctness": 0.5,
            "minimum_faithfulness": 0.95,
            "maximum_hallucination_rate": 0.05,
            "minimum_citation_precision": 0.95,
            "minimum_citation_recall": 0.95,
            "minimum_refusal_accuracy": 1.0,
            "minimum_escalation_accuracy": 1.0,
        },
    }

    checks = GATES.build_checks(
        observed,
        thresholds,
    )
    checks_by_name = {
        name: (value, direction, threshold)
        for name, value, direction, threshold in checks
    }

    assert len(checks) == 13
    assert checks_by_name["decision_accuracy"] == (
        1.0,
        "minimum",
        0.95,
    )
    assert checks_by_name["faithfulness"] == (
        1.0,
        "minimum",
        0.95,
    )
    assert checks_by_name["hallucination_rate"] == (
        0.0,
        "maximum",
        0.05,
    )
    assert checks_by_name["citation_precision"] == (
        1.0,
        "minimum",
        0.95,
    )


def test_required_metric_rejects_unavailable_values() -> None:
    try:
        GATES.required_metric(
            {"faithfulness": None},
            "faithfulness",
        )
    except ValueError as error:
        assert "faithfulness" in str(error)
    else:
        raise AssertionError(
            "Unavailable required metric was accepted."
        )