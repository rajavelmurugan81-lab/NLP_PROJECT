"""Comprehensive End-to-End Integration Test Suite for CiteGuard Pipeline (Phase 8)."""

import os
import tempfile
import pytest

from src.pipeline import CiteGuardPipeline
from src.verification.schema import VerificationResult, RetrievedEvidence, AgentDecision, AgentAssessment
from src.verification.providers.mock_provider import MockProvider
from src.evaluation.hallucination_ledger import HallucinationLedger
from src.evaluation.failure_taxonomy import FailureType


def test_integration_a_valid_scientific_claim():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        pipeline = CiteGuardPipeline(provider_type="mock", ledger_path=ledger_file)

        result = pipeline.verify(
            claim="MicroRNA-21 regulates cellular apoptosis in carcinoma models.",
            claim_id="test_a_001",
            dataset="scifact",
            mode="hybrid"
        )

        assert isinstance(result, VerificationResult)
        assert result.claim_id == "test_a_001"
        assert result.final_label in ("SUPPORT", "REFUTE", "NEI")
        assert 0.0 <= result.confidence <= 1.0
        assert result.execution_metrics.total_execution_time_sec > 0.0


def test_integration_b_empty_claim_input():
    pipeline = CiteGuardPipeline(provider_type="mock")

    result = pipeline.verify(claim="   ")  # Whitespace only
    assert result.final_label == "ERROR"
    assert result.resolution_method == "input_validation_error"
    assert "empty" in result.metadata["validation_error"].lower()


def test_integration_c_invalid_inputs():
    pipeline = CiteGuardPipeline(provider_type="mock")

    # Non-string input
    res_int = pipeline.verify(claim=12345)
    assert res_int.final_label == "ERROR"
    assert "must be a string" in res_int.metadata["validation_error"].lower()

    # Ultra-short input (< 5 chars)
    res_short = pipeline.verify(claim="DNA")
    assert res_short.final_label == "ERROR"
    assert "too short" in res_short.metadata["validation_error"].lower()

    # Ultra-long input (> 1000 chars)
    ultra_long = "MicroRNA " * 200
    res_long = pipeline.verify(claim=ultra_long)
    assert res_long.final_label == "ERROR"
    assert "too long" in res_long.metadata["validation_error"].lower()


def test_integration_d_claim_with_relevant_evidence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        pipeline = CiteGuardPipeline(
            bm25_index_path="nonexistent.pkl",
            provider_type="mock",
            ledger_path=ledger_file
        )

        result = pipeline.verify(
            claim="MicroRNA-21 regulates cellular apoptosis.",
            claim_id="test_d_001",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_001"]
        )

        assert result.final_label == "SUPPORT"
        assert result.confidence >= 0.8
        assert len(result.evidence) >= 1


def test_integration_e_claim_with_insufficient_evidence():
    pipeline = CiteGuardPipeline(bm25_index_path="nonexistent.pkl", provider_type="mock")

    # Manually pass empty evidence list to test insufficient evidence routing
    from src.verification.schema import VerificationRequest
    req = VerificationRequest(
        claim_id="test_e_001",
        claim_text="Unsubstantiated claim with no supporting literature."
    )

    result = pipeline.orchestrator.verify(request=req, pre_retrieved_evidence=[], run_id="run_e")
    assert result.final_label == "NEI"
    assert result.resolution_method == "insufficient_evidence_fallback"
    assert result.execution_metrics.agent_calls_count == 0


def test_integration_f_retrieval_failure_handling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        # Instantiate pipeline with invalid index paths
        pipeline = CiteGuardPipeline(
            bm25_index_path="invalid_path.pkl",
            faiss_index_path="invalid_path.bin",
            map_path="invalid_path.json",
            provider_type="mock",
            ledger_path=ledger_file
        )

        # Should operate safely using fallback mode without crashing
        result = pipeline.verify(claim="Scientific claim evaluated under missing index fallback.")
        assert result.final_label in ("SUPPORT", "REFUTE", "NEI")
        assert result.evidence[0].source == "fallback_mock"


def test_integration_g_agent_disagreement_and_debate():
    # Setup custom mock rule function to generate conflicting initial assessments
    def custom_conflicting_rule(prompt, sys_prompt):
        if "evidence agent" in (sys_prompt or "").lower():
            return '{"decision": "SUPPORT", "confidence": 0.9, "supporting_evidence_ids": ["doc_1"], "contradicting_evidence_ids": [], "rationale": "Evidence supports claim."}'
        elif "claim analysis agent" in (sys_prompt or "").lower():
            return '{"decision": "REFUTE", "confidence": 0.85, "supporting_evidence_ids": [], "contradicting_evidence_ids": ["doc_1"], "rationale": "Scope mismatch."}'
        elif "source quality agent" in (sys_prompt or "").lower():
            return '{"decision": "SUPPORT", "confidence": 0.7, "supporting_evidence_ids": ["doc_1"], "contradicting_evidence_ids": [], "rationale": "Source is valid."}'
        elif "verification synthesis agent" in (sys_prompt or "").lower():
            return '{"decision": "REFUTE", "confidence": 0.8, "supporting_evidence_ids": [], "contradicting_evidence_ids": ["doc_1"], "rationale": "Disagreement present."}'
        elif "debate" in (sys_prompt or "").lower() or "resolver" in (sys_prompt or "").lower():
            return '{"final_decision": "REFUTE", "consensus_reached": true, "resolution_rationale": "Debate synthesis favored refutation based on scope mismatch.", "disagreement_reasons": ["Scope mismatch"]}'
        return None

    provider = MockProvider(rule_fn=custom_conflicting_rule)
    pipeline = CiteGuardPipeline(bm25_index_path="nonexistent.pkl", provider_type="mock")
    pipeline.orchestrator.provider = provider
    pipeline.orchestrator.evidence_agent.provider = provider
    pipeline.orchestrator.claim_agent.provider = provider
    pipeline.orchestrator.source_agent.provider = provider
    pipeline.orchestrator.verification_agent.provider = provider
    pipeline.orchestrator.debate_resolver.provider = provider

    result = pipeline.verify(claim="Conflicting claim evaluated across agents.")
    assert result.disagreement_status is True
    assert result.resolution_method == "debate_resolution"
    assert result.debate_result is not None
    assert result.debate_result.final_decision.value == "REFUTE"


def test_integration_h_failure_signature_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        # Provider returns REFUTE, but ground truth is SUPPORT -> failure signature generated
        mock_provider = MockProvider(default_decision="REFUTE")
        pipeline = CiteGuardPipeline(bm25_index_path="nonexistent.pkl", provider_type="mock", ledger_path=ledger_file)
        pipeline.orchestrator.provider = mock_provider
        pipeline.orchestrator.evidence_agent.provider = mock_provider
        pipeline.orchestrator.claim_agent.provider = mock_provider
        pipeline.orchestrator.source_agent.provider = mock_provider
        pipeline.orchestrator.verification_agent.provider = mock_provider

        result = pipeline.verify(
            claim="MicroRNA-21 regulates cellular apoptosis.",
            claim_id="test_h_001",
            dataset="scifact",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_fallback_001"]
        )

        assert result.final_label == "REFUTE"
        assert result.failure_signature is not None
        assert result.failure_signature["claim_id"] == "test_h_001"
        assert result.failure_signature["failure_type"] == "REASONING_FAILURE"


def test_integration_i_hallucination_ledger_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        mock_provider = MockProvider(default_decision="REFUTE")
        pipeline = CiteGuardPipeline(bm25_index_path="nonexistent.pkl", provider_type="mock", ledger_path=ledger_file)
        pipeline.orchestrator.provider = mock_provider
        pipeline.orchestrator.evidence_agent.provider = mock_provider
        pipeline.orchestrator.claim_agent.provider = mock_provider
        pipeline.orchestrator.source_agent.provider = mock_provider
        pipeline.orchestrator.verification_agent.provider = mock_provider

        result = pipeline.verify(
            claim="MicroRNA-21 regulates cellular apoptosis.",
            claim_id="test_i_001",
            dataset="scifact",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_001"]
        )

        # Verify ledger file exists and contains the signature line
        assert os.path.exists(ledger_file)
        ledger = HallucinationLedger(ledger_filepath=ledger_file)
        sigs = ledger.signatures
        assert len(sigs) == 1
        assert sigs[0].claim_id == "test_i_001"


def test_integration_j_complete_successful_pipeline_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        pipeline = CiteGuardPipeline(bm25_index_path="nonexistent.pkl", provider_type="mock", ledger_path=ledger_file)

        result = pipeline.verify(
            claim="CRISPR-Cas9 enables targeted genome editing in eukaryotic cell lines.",
            claim_id="test_j_001",
            dataset="scifact",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_001"],
            top_k=5,
            mode="hybrid",
            run_id="run_smoke_test"
        )

        assert result.claim_id == "test_j_001"
        assert result.final_label == "SUPPORT"
        assert result.confidence > 0.0
        assert len(result.agent_assessments) in (4, 5)
        assert result.execution_metrics.agent_calls_count >= 4
        assert result.execution_metrics.total_execution_time_sec > 0.0
        assert "prompt_tokens" in result.execution_metrics.token_usage
