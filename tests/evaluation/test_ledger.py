"""Unit tests for src/evaluation/hallucination_ledger.py."""

import os
import pytest
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_signature import FailureSignature
from src.evaluation.hallucination_ledger import HallucinationLedger


def test_ledger_append_deduplication_query(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    ledger = HallucinationLedger(ledger_filepath=ledger_path)

    sig1 = FailureSignature(
        claim_id="c1", claim_text="Claim 1", dataset="scifact",
        predicted_label="REFUTE", ground_truth_label="SUPPORT",
        failure_type=FailureType.REASONING_FAILURE, run_id="run_1", retrieval_mode="hybrid"
    )

    sig2 = FailureSignature(
        claim_id="c2", claim_text="Claim 2", dataset="covert",
        predicted_label="NEUTRAL", ground_truth_label="SUPPORT",
        failure_type=FailureType.RETRIEVAL_FAILURE, run_id="run_1", retrieval_mode="bm25"
    )

    # Test append
    assert ledger.append(sig1) is True
    assert ledger.append(sig2) is True

    # Test duplicate prevention (same sig1)
    assert ledger.append(sig1) is False

    # Test disk reload
    reloaded_ledger = HallucinationLedger(ledger_filepath=ledger_path)
    assert len(reloaded_ledger.signatures) == 2

    # Test querying
    scifact_results = reloaded_ledger.query(dataset="scifact")
    assert len(scifact_results) == 1
    assert scifact_results[0].claim_id == "c1"

    bm25_results = reloaded_ledger.query(retrieval_mode="bm25")
    assert len(bm25_results) == 1
    assert bm25_results[0].claim_id == "c2"


def test_ledger_export_csv(tmp_path):
    ledger_path = str(tmp_path / "ledger.jsonl")
    csv_path = str(tmp_path / "ledger.csv")
    ledger = HallucinationLedger(ledger_filepath=ledger_path)

    sig = FailureSignature(
        claim_id="c1", claim_text="Claim 1", dataset="scifact",
        predicted_label="REFUTE", ground_truth_label="SUPPORT",
        failure_type=FailureType.REASONING_FAILURE
    )
    ledger.append(sig)

    exported_csv = ledger.export_csv(csv_filepath=csv_path)
    assert os.path.exists(exported_csv)
    with open(exported_csv, "r", encoding="utf-8") as f:
        content = f.read()
        assert "signature_hash" in content
        assert "REASONING_FAILURE" in content
