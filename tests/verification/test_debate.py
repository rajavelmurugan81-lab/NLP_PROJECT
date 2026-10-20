"""Unit tests for DebateResolver synthesis."""

import pytest
from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence
from src.verification.providers.mock_provider import MockProvider
from src.verification.debate import DebateResolver


def test_debate_resolver_execution():
    mock_provider = MockProvider(default_decision="REFUTE")
    resolver = DebateResolver(provider=mock_provider)

    evidence = [
        RetrievedEvidence(doc_id="doc_1", text="Contradictory passage evidence.", score=0.9)
    ]

    conflicting_assessments = [
        AgentAssessment(agent_name="A1", decision=AgentDecision.SUPPORT, confidence=0.7),
        AgentAssessment(agent_name="A2", decision=AgentDecision.REFUTE, confidence=0.85),
    ]

    result = resolver.resolve(
        claim_text="Test conflicting claim",
        evidence=evidence,
        assessments=conflicting_assessments
    )

    assert result.rounds == 1
    assert result.final_decision == AgentDecision.REFUTE
    assert len(result.revised_assessments) == 2
