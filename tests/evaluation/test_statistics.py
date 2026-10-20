"""Unit tests for src/evaluation/failure_statistics.py."""

import os
import pytest
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_signature import FailureSignature
from src.evaluation.hallucination_ledger import HallucinationLedger
from src.evaluation.failure_statistics import (
    compute_failure_statistics,
    generate_visualization_tables
)


def test_compute_failure_statistics():
    sig1 = FailureSignature(
        claim_id="c1", claim_text="T1", dataset="scifact",
        predicted_label="REFUTE", ground_truth_label="SUPPORT",
        failure_type=FailureType.RETRIEVAL_FAILURE, retrieval_mode="bm25"
    )
    sig2 = FailureSignature(
        claim_id="c2", claim_text="T2", dataset="scifact",
        predicted_label="NEUTRAL", ground_truth_label="SUPPORT",
        failure_type=FailureType.REASONING_FAILURE, retrieval_mode="hybrid"
    )

    stats = compute_failure_statistics([sig1, sig2], total_evaluated_cases=10)

    assert stats["total_evaluated_cases"] == 10
    assert stats["incorrect_cases"] == 2
    assert stats["correct_cases"] == 8
    assert stats["failure_percentage"] == 20.0
    assert stats["specific_counts"]["retrieval_failures"] == 1
    assert stats["specific_counts"]["reasoning_failures"] == 1


def test_generate_visualization_tables(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    out_dir = str(tmp_path / "viz")
    ledger = HallucinationLedger(ledger_filepath=ledger_path)

    sig = FailureSignature(
        claim_id="c1", claim_text="T1", dataset="scifact",
        predicted_label="REFUTE", ground_truth_label="SUPPORT",
        failure_type=FailureType.RETRIEVAL_FAILURE
    )
    ledger.append(sig)

    files = generate_visualization_tables(ledger, output_dir=out_dir)
    assert len(files) == 4
    for fname, fpath in files.items():
        assert os.path.exists(fpath)
