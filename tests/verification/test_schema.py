"""Unit tests for verification schemas and data structures."""

import pytest
from src.verification.schema import (
    AgentDecision,
    VerificationRequest,
    RetrievedEvidence,
    AgentAssessment,
    DebateResult,
    ExecutionMetrics,
    VerificationResult
)


def test_agent_decision_enum():
    assert AgentDecision.from_str("SUPPORT") == AgentDecision.SUPPORT
    assert AgentDecision.from_str("supported") == AgentDecision.SUPPORT
    assert AgentDecision.from_str("REFUTE") == AgentDecision.REFUTE
    assert AgentDecision.from_str("refuted") == AgentDecision.REFUTE
    assert AgentDecision.from_str("NEI") == AgentDecision.NEI
    assert AgentDecision.from_str("neutral") == AgentDecision.NEI
    assert AgentDecision.from_str("invalid_str") == AgentDecision.UNKNOWN


def test_retrieved_evidence_serialization():
    ev = RetrievedEvidence(
        doc_id="doc_101",
        text="Sample scientific text passage.",
        score=0.95,
        rank=1,
        source="bm25",
        title="Sample Paper Title"
    )
    d = ev.to_dict()
    assert d["doc_id"] == "doc_101"
    assert d["score"] == 0.95
    assert d["rank"] == 1


def test_agent_assessment_clamping():
    assessment = AgentAssessment(
        agent_name="TestAgent",
        decision="SUPPORT",
        confidence=1.5,  # Out of bounds -> should clamp to 1.0
        uncertainty=-0.2  # Out of bounds -> should clamp to 0.0
    )
    assert assessment.confidence == 1.0
    assert assessment.uncertainty == 0.0
    assert assessment.decision == AgentDecision.SUPPORT

    d = assessment.to_dict()
    assert d["decision"] == "SUPPORT"


def test_verification_result_serialization():
    ev = RetrievedEvidence(doc_id="doc_1", text="Evidence passage", score=0.8)
    a = AgentAssessment(agent_name="Agent1", decision=AgentDecision.SUPPORT, confidence=0.85)
    metrics = ExecutionMetrics(total_execution_time_sec=1.23, agent_calls_count=4)

    res = VerificationResult(
        claim_id="c_001",
        claim_text="Test Claim",
        final_label="SUPPORT",
        confidence=0.85,
        evidence=[ev],
        agent_assessments=[a],
        execution_metrics=metrics
    )

    d = res.to_dict()
    assert d["claim_id"] == "c_001"
    assert d["final_label"] == "SUPPORT"
    assert len(d["evidence"]) == 1
    assert d["execution_metrics"]["agent_calls_count"] == 4

    json_str = res.to_json()
    assert '"claim_id": "c_001"' in json_str
