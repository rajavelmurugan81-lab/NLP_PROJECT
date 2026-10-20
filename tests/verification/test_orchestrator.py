"""Unit tests for MultiAgentOrchestrator and Phase 5 Ledger integration."""

import os
import tempfile
import pytest

from src.verification.schema import VerificationRequest, RetrievedEvidence, AgentDecision
from src.verification.providers.mock_provider import MockProvider
from src.verification.orchestrator import MultiAgentOrchestrator
from src.evaluation.hallucination_ledger import HallucinationLedger


def test_orchestrator_consensus_flow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        
        mock_provider = MockProvider(default_decision="SUPPORT", default_confidence=0.9)
        orchestrator = MultiAgentOrchestrator(
            provider=mock_provider,
            ledger_path=ledger_file
        )

        request = VerificationRequest(
            claim_id="claim_001",
            claim_text="MicroRNA-21 regulates apoptosis in cancer cells.",
            dataset="scifact",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_001"]
        )

        pre_retrieved = [
            RetrievedEvidence(doc_id="doc_001", text="Study confirms miR-21 regulates apoptosis.", score=0.92, rank=1)
        ]

        result = orchestrator.verify(request, pre_retrieved_evidence=pre_retrieved, run_id="test_run_1")

        assert result.claim_id == "claim_001"
        assert result.final_label == "SUPPORT"
        assert result.confidence >= 0.8
        assert not result.disagreement_status
        assert result.resolution_method == "direct_consensus"
        assert result.execution_metrics.agent_calls_count in (4, 5)
        assert any(a.agent_name == "SkepticAgent" for a in result.agent_assessments)


def test_orchestrator_failure_ledger_integration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        
        # Provider returns REFUTE, but ground truth is SUPPORT -> should trigger failure signature & write to ledger
        mock_provider = MockProvider(default_decision="REFUTE", default_confidence=0.85)
        orchestrator = MultiAgentOrchestrator(
            provider=mock_provider,
            ledger_path=ledger_file
        )

        request = VerificationRequest(
            claim_id="claim_002",
            claim_text="MicroRNA-21 regulates apoptosis in cancer cells.",
            dataset="scifact",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_001"]
        )

        pre_retrieved = [
            RetrievedEvidence(doc_id="doc_001", text="Study confirms miR-21 regulates apoptosis.", score=0.92, rank=1)
        ]

        result = orchestrator.verify(request, pre_retrieved_evidence=pre_retrieved, run_id="test_run_2")

        assert result.final_label == "REFUTE"
        assert result.failure_signature is not None
        assert result.failure_signature["claim_id"] == "claim_002"
        assert result.failure_signature["failure_type"] == "REASONING_FAILURE"

        # Verify record was appended to persistent ledger
        ledger = HallucinationLedger(ledger_filepath=ledger_file)
        signatures = ledger.signatures
        assert len(signatures) == 1
        assert signatures[0].claim_id == "claim_002"
        assert signatures[0].failure_type.value == "REASONING_FAILURE"


def test_orchestrator_insufficient_evidence_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        orchestrator = MultiAgentOrchestrator(
            provider=MockProvider(),
            ledger_path=ledger_file
        )

        request = VerificationRequest(
            claim_id="claim_003",
            claim_text="Unsubstantiated claim with no evidence."
        )

        # Empty evidence -> trigger insufficient evidence fallback
        result = orchestrator.verify(request, pre_retrieved_evidence=[], run_id="test_run_3")

        assert result.final_label == "NEI"
        assert result.resolution_method == "insufficient_evidence_fallback"
        assert result.execution_metrics.agent_calls_count == 0
