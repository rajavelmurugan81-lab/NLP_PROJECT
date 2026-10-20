"""Unit tests for CiteGuard SkepticAgent and Adversarial Verification Flow."""

import pytest
from typing import List, Dict, Any

from src.verification.schema import VerificationRequest, RetrievedEvidence, AgentDecision
from src.verification.providers.mock_provider import MockProvider
from src.verification.agents.verification_agent import VerificationAgent
from src.verification.skeptic_agent import SkepticAgent, SkepticResult
from src.evaluation.failure_taxonomy import CanonicalFailureCode
from src.evaluation.failure_classifier import FailureClassifier
from src.verification.orchestrator import MultiAgentOrchestrator


def test_skeptic_clean_supported_claim():
    """Verify clean supported claim results in NO_OBJECTION and SUPPORT decision."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_clean_1",
            text="Empirical results demonstrate that microRNA-21 downregulates PTEN in glioblastoma cell cultures.",
            score=0.92,
            rank=1,
            metadata={"relationship": "SUPPORT", "publication_year": 2023, "doi": "10.1038/s41586-023-0001"}
        )
    ]
    res = agent.evaluate(
        claim_text="MicroRNA-21 downregulates PTEN in glioblastoma cell lines.",
        evidence=evidence,
        current_verdict="SUPPORT",
        confidence=0.88
    )
    assert not res.challenged
    assert res.objection_level == "NO_OBJECTION"
    assert res.primary_failure_code is None
    assert len(res.objections) == 0

    assessment = agent.analyze({
        "claim_text": "MicroRNA-21 downregulates PTEN in glioblastoma cell lines.",
        "evidence": evidence
    })
    assert assessment.decision == AgentDecision.SUPPORT
    assert assessment.confidence >= 0.80
    assert "skeptic_result" in assessment.metadata


def test_skeptic_f1_unsupported_claim():
    """Verify empty evidence or low scores trigger F1 Unsupported Claim."""
    agent = SkepticAgent()
    
    # Empty evidence
    res_empty = agent.evaluate(
        claim_text="Quantum gravity unified with dark energy.",
        evidence=[],
        current_verdict="SUPPORT"
    )
    assert res_empty.challenged
    assert res_empty.objection_level == "CONFIRMED_FAILURE"
    assert res_empty.primary_failure_code == CanonicalFailureCode.F1

    # Subthreshold score (< 0.70) with SUPPORT verdict
    evidence_low = [
        RetrievedEvidence(
            doc_id="doc_low",
            text="Vague correlation observed in preliminary simulation.",
            score=0.45,
            rank=1,
            metadata={"relationship": "NEUTRAL"}
        )
    ]
    res_low = agent.evaluate(
        claim_text="Quantum gravity unified with dark energy.",
        evidence=evidence_low,
        current_verdict="SUPPORT"
    )
    assert res_low.challenged
    assert res_low.objection_level == "CONFIRMED_FAILURE"
    assert res_low.primary_failure_code == CanonicalFailureCode.F1


def test_skeptic_f2_citation_mismatch():
    """Verify unverified bibliographic citation triggers F2 Citation Mismatch."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_f2",
            text="Theoretical bounds on algorithmic stability.",
            score=0.85,
            rank=1,
            metadata={"citation_mismatch": True, "doi": ""}
        )
    ]
    res = agent.evaluate(
        claim_text="Stability bounds established in Paper 2024.",
        evidence=evidence,
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F2
    assert "F2 - Citation Mismatch" in res.objections[0]


def test_skeptic_f3_overgeneralization():
    """Verify universal quantifiers with narrow evidence trigger F3 Overgeneralization."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_f3",
            text="Observed efficacy in a sample of 12 female mice.",
            score=0.88,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    # Universal claim
    res = agent.evaluate(
        claim_text="The treatment universally cures all human populations always.",
        evidence=evidence,
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F3
    assert any("Overgeneralization" in obj for obj in res.objections)


def test_skeptic_f4_causal_leap():
    """Verify direct causal wording on correlational evidence triggers F4 Causal Leap."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_f4",
            text="In an observational cohort of 500 patients, dietary sodium was correlated with hypertension biomarkers.",
            score=0.88,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    # Causal claim on correlational evidence
    res = agent.evaluate(
        claim_text="Dietary sodium causes hypertension and directly proves cardiovascular pathology.",
        evidence=evidence,
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F4
    assert any("Causal Leap" in obj for obj in res.objections)


def test_skeptic_f5_outdated_evidence():
    """Verify superseded or outdated evidence triggers F5 Outdated Evidence."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_f5",
            text="Early 2005 survey of neural network backpropagation limits.",
            score=0.82,
            rank=1,
            metadata={"publication_year": 2005, "is_superseded": True}
        )
    ]
    res = agent.evaluate(
        claim_text="Modern LLM transformer training in 2026 relies on early backprop bounds.",
        evidence=evidence,
        question="What are modern transformer training methods in 2026?",
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F5
    assert any("Outdated Evidence" in obj for obj in res.objections)


def test_skeptic_f6_population_context_mismatch():
    """Verify off-domain evidence triggers F6 Population / Context Mismatch."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_f6",
            text="Smart grid voltage stability and electric distribution network optimization.",
            score=0.78,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    res = agent.evaluate(
        claim_text="Quantum federated learning accelerates convergence across distributed nodes.",
        evidence=evidence,
        question="How does quantum federated learning scale?",
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F6
    assert any("Context Mismatch" in obj for obj in res.objections)


def test_skeptic_f7_contradictory_evidence():
    """Verify contradictory evidence triggers F7 Contradictory Evidence with highest hierarchy."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_contra",
            text="Clinical trial conclusively demonstrates drug X showed NO benefit over placebo and induced cardiotoxicity.",
            score=0.91,
            rank=1,
            metadata={"relationship": "REFUTE"}
        )
    ]
    res = agent.evaluate(
        claim_text="Drug X provides robust clinical benefit for cardiovascular therapy.",
        evidence=evidence,
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F7
    assert any("Contradictory Evidence" in obj for obj in res.objections)

    # SkepticAgent analyze returns REFUTE when F7 confirmed
    assessment = agent.analyze({
        "claim_text": "Drug X provides robust clinical benefit for cardiovascular therapy.",
        "evidence": evidence
    })
    assert assessment.decision == AgentDecision.REFUTE


def test_skeptic_f8_evidence_misinterpretation():
    """Verify syntactic claim malformation triggers F8 Evidence Misinterpretation."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_f8",
            text="Gene expression modulation during cellular senescence.",
            score=0.85,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    res = agent.evaluate(
        claim_text="Target protein altering increasing cellular survival directly affects target scientific outcomes.",
        evidence=evidence,
        current_verdict="SUPPORT"
    )
    assert res.challenged
    assert res.objection_level == "CONFIRMED_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F8


def test_skeptic_potential_vs_confirmed():
    """Verify weak evidence without SUPPORT verdict produces POTENTIAL_FAILURE."""
    agent = SkepticAgent()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_weak",
            text="Inconclusive early benchmark results.",
            score=0.48,
            rank=1,
            metadata={"relationship": "NEUTRAL"}
        )
    ]
    res = agent.evaluate(
        claim_text="Preliminary speculative hypothesis.",
        evidence=evidence,
        current_verdict="NEI",
        confidence=0.50
    )
    assert res.challenged
    assert res.objection_level == "POTENTIAL_FAILURE"
    assert res.primary_failure_code == CanonicalFailureCode.F1


def test_verification_agent_revises_verdict_on_skeptic_objection():
    """Verify VerificationAgent inspects skeptic_result and downgrades or revises decision."""
    provider = MockProvider(default_decision="SUPPORT", default_confidence=0.90)
    verifier = VerificationAgent(provider=provider)

    # Case 1: Skeptic confirms F7 contradiction -> verifier switches to REFUTE
    context_f7 = {
        "claim_text": "Drug X cures disease Y.",
        "evidence": [],
        "skeptic_result": {
            "objection_level": "CONFIRMED_FAILURE",
            "primary_failure_code": CanonicalFailureCode.F7,
            "objections": ["Contradictory clinical trial evidence refutes claim."]
        }
    }
    assessment_f7 = verifier.analyze(context_f7)
    assert assessment_f7.decision == AgentDecision.REFUTE
    assert "Skeptic Revision" in assessment_f7.rationale

    # Case 2: Skeptic confirms F1 unsupported -> verifier switches to NEI
    context_f1 = {
        "claim_text": "Unsubstantiated phenomenon.",
        "evidence": [],
        "skeptic_result": {
            "objection_level": "CONFIRMED_FAILURE",
            "primary_failure_code": CanonicalFailureCode.F1,
            "objections": ["No evidence retrieved."]
        }
    }
    assessment_f1 = verifier.analyze(context_f1)
    assert assessment_f1.decision == AgentDecision.NEI
    assert "Skeptic Revision" in assessment_f1.rationale

    # Case 3: Skeptic confirms F4 causal leap -> verifier switches to PARTIAL
    context_f4 = {
        "claim_text": "A causes B.",
        "evidence": [],
        "skeptic_result": {
            "objection_level": "CONFIRMED_FAILURE",
            "primary_failure_code": CanonicalFailureCode.F4,
            "objections": ["Correlation asserted as causation."]
        }
    }
    assessment_f4 = verifier.analyze(context_f4)
    assert assessment_f4.decision == AgentDecision.PARTIAL
    assert "Skeptic Revision" in assessment_f4.rationale


def test_failure_classifier_incorporates_skeptic_result():
    """Verify FailureClassifier honors Skeptic's confirmed failure and diagnostic signals."""
    classifier = FailureClassifier()
    evidence = [
        RetrievedEvidence(
            doc_id="doc_fc",
            text="Observational correlation between biomarker and outcome.",
            score=0.82,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    skeptic_res = SkepticResult(
        challenged=True,
        objection_level="CONFIRMED_FAILURE",
        failure_candidates=[CanonicalFailureCode.F4],
        primary_failure_code=CanonicalFailureCode.F4,
        primary_failure_name="Causal Leap",
        confidence=0.88,
        objections=["Claim asserts direct causality on correlational evidence."],
        recommended_action="DEMOTE_CAUSAL_TO_CORRELATIONAL"
    )

    res = classifier.classify(
        claim_text="Biomarker causes therapeutic outcome.",
        evidence=evidence,
        verdict="SUPPORT",
        confidence=0.85,
        skeptic_result=skeptic_res
    )
    assert res.has_failure
    assert res.canonical_code == CanonicalFailureCode.F4
    assert res.diagnostic_signals["skeptic_objection_level"] == "CONFIRMED_FAILURE"
    assert "F4" in res.diagnostic_signals["skeptic_failure_candidates"]


def test_orchestrator_runs_skeptic_in_sequence():
    """Verify MultiAgentOrchestrator runs 5 agents in proper sequence and outputs skeptic metadata."""
    mock_provider = MockProvider(default_decision="SUPPORT", default_confidence=0.90)
    orchestrator = MultiAgentOrchestrator(provider=mock_provider)

    evidence = [
        RetrievedEvidence(
            doc_id="doc_orch_1",
            text="miR-21 regulation of apoptosis confirmed in cell assays.",
            score=0.92,
            rank=1,
            metadata={"relationship": "SUPPORT", "doi": "10.1000/182"}
        )
    ]
    request = VerificationRequest(
        claim_id="claim_orch_01",
        claim_text="miR-21 regulates apoptosis in cell assays."
    )

    result = orchestrator.verify(request, pre_retrieved_evidence=evidence, run_id="test_run_skeptic")
    assert result.execution_metrics.agent_calls_count in (4, 5)

    agent_names = [a.agent_name for a in result.agent_assessments]
    assert "EvidenceAgent" in agent_names
    assert "ClaimAnalysisAgent" in agent_names
    assert "SourceQualityAgent" in agent_names
    assert "SkepticAgent" in agent_names
    assert "VerificationAgent" in agent_names

    # Check SkepticAgent ran before VerificationAgent
    skeptic_idx = agent_names.index("SkepticAgent")
    verif_idx = agent_names.index("VerificationAgent")
    assert skeptic_idx < verif_idx

    # Check skeptic_result is embedded in result metadata
    assert "skeptic_result" in result.metadata
    assert result.metadata["skeptic_result"]["objection_level"] == "NO_OBJECTION"
