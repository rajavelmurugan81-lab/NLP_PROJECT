"""Unit tests for src/evaluation/failure_signature.py."""

import pytest
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_signature import FailureSignature


def test_failure_signature_creation_and_hashing():
    sig = FailureSignature(
        claim_id="claim_001",
        claim_text="CRISPR edits DNA.",
        dataset="scifact",
        predicted_label="REFUTE",
        ground_truth_label="SUPPORT",
        failure_type=FailureType.REASONING_FAILURE,
        run_id="run_test_01"
    )

    assert sig.claim_id == "claim_001"
    assert sig.failure_type == FailureType.REASONING_FAILURE
    h1 = sig.compute_signature_hash()
    assert isinstance(h1, str)
    assert len(h1) == 16


def test_failure_signature_serialization():
    sig = FailureSignature(
        claim_id="claim_002",
        claim_text="Sample claim",
        dataset="covert",
        predicted_label="NEUTRAL",
        ground_truth_label="SUPPORT",
        failure_type=FailureType.RETRIEVAL_FAILURE
    )

    d = sig.to_dict()
    assert d["failure_type"] == "RETRIEVAL_FAILURE"

    recon = FailureSignature.from_dict(d)
    assert recon.claim_id == sig.claim_id
    assert recon.failure_type == FailureType.RETRIEVAL_FAILURE


def test_failure_signature_invalid_type_fallback():
    d = {
        "claim_id": "c3",
        "claim_text": "Text",
        "dataset": "scifact",
        "predicted_label": "X",
        "ground_truth_label": "Y",
        "failure_type": "INVALID_UNKNOWN_TYPE"
    }
    sig = FailureSignature.from_dict(d)
    assert sig.failure_type == FailureType.UNKNOWN_FAILURE
