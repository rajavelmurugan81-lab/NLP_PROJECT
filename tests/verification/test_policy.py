"""Unit tests for adaptive verification policy and agreement routing."""

import pytest
from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence
from src.verification.policy import VerificationPolicy


def test_policy_consensus_detection():
    policy = VerificationPolicy(confidence_threshold=0.7, agreement_threshold=0.75)

    assessments = [
        AgentAssessment(agent_name="A1", decision=AgentDecision.SUPPORT, confidence=0.9),
        AgentAssessment(agent_name="A2", decision=AgentDecision.SUPPORT, confidence=0.85),
        AgentAssessment(agent_name="A3", decision=AgentDecision.SUPPORT, confidence=0.88),
        AgentAssessment(agent_name="A4", decision=AgentDecision.SUPPORT, confidence=0.92),
    ]

    agreement = policy.evaluate_agreement(assessments)
    assert agreement["agreement_ratio"] == 1.0
    assert agreement["majority_decision"] == AgentDecision.SUPPORT
    assert not agreement["disagreement_status"]
    assert not policy.should_trigger_debate(assessments)


def test_policy_disagreement_detection():
    policy = VerificationPolicy(confidence_threshold=0.7, agreement_threshold=0.75)

    assessments = [
        AgentAssessment(agent_name="A1", decision=AgentDecision.SUPPORT, confidence=0.9),
        AgentAssessment(agent_name="A2", decision=AgentDecision.REFUTE, confidence=0.85),
        AgentAssessment(agent_name="A3", decision=AgentDecision.SUPPORT, confidence=0.8),
        AgentAssessment(agent_name="A4", decision=AgentDecision.NEI, confidence=0.7),
    ]

    agreement = policy.evaluate_agreement(assessments)
    assert agreement["agreement_ratio"] == 0.5  # 2/4 SUPPORT
    assert agreement["disagreement_status"]
    assert policy.should_trigger_debate(assessments)


def test_policy_evidence_sufficiency():
    policy = VerificationPolicy(evidence_quality_threshold=0.5)

    strong_evidence = [RetrievedEvidence(doc_id="doc_1", text="Text", score=0.85)]
    weak_evidence = [RetrievedEvidence(doc_id="doc_1", text="Text", score=0.15)]
    empty_evidence = []

    assert policy.is_evidence_sufficient(strong_evidence)
    assert not policy.is_evidence_sufficient(weak_evidence)
    assert not policy.is_evidence_sufficient(empty_evidence)
