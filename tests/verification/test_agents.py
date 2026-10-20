"""Unit tests for verification agent roles."""

import pytest
from src.verification.schema import AgentDecision, RetrievedEvidence, VerificationRequest
from src.verification.providers.mock_provider import MockProvider
from src.verification.agents.evidence_agent import EvidenceAgent
from src.verification.agents.claim_analysis_agent import ClaimAnalysisAgent
from src.verification.agents.source_quality_agent import SourceQualityAgent
from src.verification.agents.verification_agent import VerificationAgent


def test_agent_roles_execution():
    mock_provider = MockProvider(default_decision="SUPPORT", default_confidence=0.88)
    
    evidence_agent = EvidenceAgent(provider=mock_provider)
    claim_agent = ClaimAnalysisAgent(provider=mock_provider)
    source_agent = SourceQualityAgent(provider=mock_provider)
    verifier_agent = VerificationAgent(provider=mock_provider)

    evidence = [
        RetrievedEvidence(doc_id="doc_1", text="Sample evidence passage supporting the claim.", score=0.9)
    ]
    context = {
        "claim_text": "MicroRNA-21 inhibits cell apoptosis.",
        "evidence": evidence
    }

    res_ev = evidence_agent.analyze(context)
    assert res_ev.agent_name == "EvidenceAgent"
    assert res_ev.decision == AgentDecision.SUPPORT

    res_cl = claim_agent.analyze(context)
    assert res_cl.agent_name == "ClaimAnalysisAgent"

    res_sq = source_agent.analyze(context)
    assert res_sq.agent_name == "SourceQualityAgent"

    res_vf = verifier_agent.analyze(context)
    assert res_vf.agent_name == "VerificationAgent"
    assert res_vf.confidence == 0.88
