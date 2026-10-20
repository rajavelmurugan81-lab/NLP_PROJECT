"""Tests for Runtime Hallucination Audit Ledger in CiteGuard."""

import os
import json
import pytest
from src.evaluation.runtime_auditor import RuntimeAuditor, RuntimeAuditRecord


def test_runtime_auditor_clean_claim(tmp_path):
    ledger_file = str(tmp_path / "runtime_audit_test.jsonl")
    auditor = RuntimeAuditor(ledger_filepath=ledger_file)
    
    mock_evidence = [
        {"score": 0.88, "relation": "supports", "metadata": {"relationship": "SUPPORTS"}},
        {"score": 0.82, "relation": "supports", "metadata": {"relationship": "SUPPORTS"}}
    ]
    
    record = auditor.audit_claim(
        claim_id="c_001",
        claim_text="Sleep deprivation reduces cognitive performance.",
        verdict="SUPPORTED",
        confidence=0.85,
        evidence=mock_evidence,
        question="Does sleep improve cognitive performance?",
        run_id="test_run_1"
    )
    
    assert record.failure_type == "NONE"
    assert record.failure_signature == "NONE"
    assert record.severity == "NONE"
    assert record.ledger_status == "CLEAN"
    assert os.path.exists(ledger_file)
    
    records = auditor.load_records()
    assert len(records) == 1
    assert records[0]["claim_id"] == "c_001"


def test_runtime_auditor_ungrounded_support(tmp_path):
    ledger_file = str(tmp_path / "runtime_audit_test.jsonl")
    auditor = RuntimeAuditor(ledger_filepath=ledger_file)
    
    # Evidence score below 0.70 threshold for SUPPORTED
    mock_evidence = [
        {"score": 0.45, "relation": "partially_supports", "metadata": {"relationship": "PARTIALLY_SUPPORTS"}}
    ]
    
    record = auditor.audit_claim(
        claim_id="c_002",
        claim_text="A complex claim without strong evidence.",
        verdict="SUPPORTED",
        confidence=0.80,
        evidence=mock_evidence,
        question="Test Question",
        run_id="test_run_2"
    )
    
    assert record.failure_type == "UNGROUNDED_SUPPORT"
    assert record.severity == "HIGH"
    assert record.ledger_status == "LOGGED_TO_AUDIT"


def test_runtime_auditor_no_evidence(tmp_path):
    ledger_file = str(tmp_path / "runtime_audit_test.jsonl")
    auditor = RuntimeAuditor(ledger_filepath=ledger_file)
    
    record = auditor.audit_claim(
        claim_id="c_003",
        claim_text="Unverifiable future assertion.",
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.35,
        evidence=[],
        question="Future Question",
        run_id="test_run_3"
    )
    
    assert record.failure_type == "NO_EVIDENCE"
    assert record.severity == "MEDIUM"
    assert record.ledger_status == "LOGGED_TO_AUDIT"


def test_runtime_auditor_contradiction(tmp_path):
    ledger_file = str(tmp_path / "runtime_audit_test.jsonl")
    auditor = RuntimeAuditor(ledger_filepath=ledger_file)
    
    mock_evidence = [
        {"score": 0.85, "relation": "contradicts", "metadata": {"relationship": "CONTRADICTS"}}
    ]
    
    record = auditor.audit_claim(
        claim_id="c_004",
        claim_text="Claim contradicted by evidence.",
        verdict="SUPPORTED",
        confidence=0.85,
        evidence=mock_evidence,
        question="Contradiction Question",
        run_id="test_run_4"
    )
    
    assert record.failure_type == "CONTRADICTORY_EVIDENCE"
    assert record.severity == "CRITICAL"
    assert record.ledger_status == "LOGGED_TO_AUDIT"
