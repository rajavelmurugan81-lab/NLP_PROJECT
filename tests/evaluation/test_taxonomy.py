"""Unit tests for src/evaluation/failure_taxonomy.py."""

import pytest
from src.evaluation.failure_taxonomy import FailureType, get_failure_type_description


def test_failure_type_enum_values():
    assert FailureType.NO_EVIDENCE.value == "NO_EVIDENCE"
    assert FailureType.RETRIEVAL_FAILURE.value == "RETRIEVAL_FAILURE"
    assert FailureType.REASONING_FAILURE.value == "REASONING_FAILURE"
    assert FailureType.UNKNOWN_FAILURE.value == "UNKNOWN_FAILURE"


def test_get_failure_type_description():
    desc = get_failure_type_description(FailureType.RETRIEVAL_FAILURE)
    assert "Ground-truth evidence exists" in desc or "retrieved" in desc
