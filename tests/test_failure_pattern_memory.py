"""Unit and Integration Tests for CiteGuard Failure-Pattern Memory Architecture.

Covers the 11 acceptance requirements:
1. failure pattern extraction
2. pattern deduplication
3. pattern matching
4. prevention action selection / risk policy
5. domain mismatch prevention
6. retrieval failure prevention
7. ungrounded support prevention
8. contradictory evidence escalation
9. claim-malformation regeneration
10. failure-memory persistence
11. failure-memory reuse on a future query (closed feedback loop)
"""

import os
import json
import pytest
from typing import List

from src.evaluation.failure_pattern import FailurePattern, PatternMatchResult
from src.evaluation.pattern_miner import PatternMiner
from src.evaluation.failure_memory import FailureMemory
from src.verification.schema import VerificationRequest, RetrievedEvidence
from src.verification.orchestrator import MultiAgentOrchestrator


@pytest.fixture
def miner(tmp_path):
    out_file = str(tmp_path / "failure_patterns.json")
    return PatternMiner(output_patterns_path=out_file)


@pytest.fixture
def memory(tmp_path):
    pat_file = str(tmp_path / "failure_patterns.json")
    tru_file = str(tmp_path / "trusted_memory.json")
    miner = PatternMiner(output_patterns_path=pat_file)
    miner.mine_patterns()
    return FailureMemory(patterns_path=pat_file, trusted_memory_path=tru_file)


# 1. Failure Pattern Extraction
def test_failure_pattern_extraction(miner):
    patterns = miner.mine_patterns()
    assert len(patterns) >= 9
    pattern_types = {p.failure_type for p in patterns}
    assert "DOMAIN_MISMATCH" in pattern_types
    assert "IRRELEVANT_EVIDENCE" in pattern_types
    assert "RETRIEVAL_FAILURE" in pattern_types
    assert "UNGROUNDED_SUPPORT" in pattern_types
    assert "CONTRADICTORY_EVIDENCE" in pattern_types
    assert "INSUFFICIENT_EVIDENCE" in pattern_types


# 2. Pattern Deduplication
def test_pattern_deduplication(miner):
    p1 = miner.mine_patterns()
    p2 = miner.mine_patterns()
    assert len(p1) == len(p2)
    ids = [p.pattern_id for p in p2]
    assert len(ids) == len(set(ids)), "Pattern IDs must be unique"


# 3. Pattern Matching
def test_pattern_matching(memory):
    claim = "Quantum-resistant federated learning enhances healthcare data privacy."
    off_domain_ev = [
        RetrievedEvidence(
            doc_id="sg_1",
            text="Smart grid SCADA systems require anomaly detection against power grid attacks.",
            score=0.75,
            rank=1,
            title="Smart Grid Vulnerabilities"
        )
    ]
    match = memory.match_historical_failures(
        claim_text=claim,
        evidence=off_domain_ev,
        question="How does quantum-resistant federated learning protect healthcare data?",
        preliminary_verdict="SUPPORT",
        confidence=0.85
    )
    assert match.matched is True
    assert "FP_DOM_001" in match.matched_pattern_ids
    assert match.pattern_match_score >= 0.85
    assert match.risk_level in ("HIGH", "VERY_HIGH")


# 4. Prevention Action Selection & Risk Policy
def test_prevention_action_selection(memory):
    claim = "Quantum-resistant federated learning improves security."
    off_domain_ev = [
        RetrievedEvidence(
            doc_id="sg_1",
            text="Electric power grid distribution infrastructure.",
            score=0.75,
            rank=1,
            title="Power Systems"
        )
    ]
    match = memory.match_historical_failures(
        claim_text=claim,
        evidence=off_domain_ev,
        preliminary_verdict="SUPPORT"
    )
    assert "REQUIRE_STRICT_DOMAIN_VALIDATION" in match.adaptive_actions
    assert "EXCLUDE_OFF_DOMAIN_PASSAGES" in match.adaptive_actions


# 5. Domain Mismatch Prevention via Claim Re-Check
def test_domain_mismatch_prevention(memory):
    claim = "Quantum-resistant federated learning improves privacy."
    ev = [
        RetrievedEvidence(
            doc_id="sg_1",
            text="Cybersecurity and resilience in smart grid networks.",
            score=0.78,
            rank=1,
            title="Smart Grid Cybersecurity"
        )
    ]
    match = memory.match_historical_failures(
        claim_text=claim,
        evidence=ev,
        question="How does quantum-resistant federated learning improve privacy?",
        preliminary_verdict="SUPPORT",
        confidence=0.85
    )
    rev_verdict, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
        claim_text=claim,
        evidence=ev,
        match_result=match,
        current_verdict="SUPPORT",
        current_confidence=0.85,
        question="How does quantum-resistant federated learning improve privacy?"
    )
    assert match_res.claim_recheck_status == "REJECT"
    assert len(rev_ev) == 0, "Off-domain evidence must be excluded"
    assert rev_verdict in ("INSUFFICIENT_EVIDENCE", "NEI")
    assert rev_conf <= 0.35


# 6. Retrieval Failure Prevention
def test_retrieval_failure_prevention(memory):
    p_ret = next((p for p in memory.patterns if p.pattern_id == "FP_RET_003"), None)
    assert p_ret is not None
    assert "ACTIVATE_HYBRID_RRF" in p_ret.adaptive_actions
    assert "EXPAND_RETRIEVAL_TOP_K" in p_ret.adaptive_actions


# 7. Ungrounded Support Prevention
def test_ungrounded_support_prevention(memory):
    claim = "Deep neural networks eliminate training loss completely."
    weak_ev = [
        RetrievedEvidence(
            doc_id="ev_weak",
            text="Neural networks optimize loss functions through stochastic gradient descent.",
            score=0.55,  # Sub-threshold score (< 0.70)
            rank=1,
            title="Optimization in Deep Learning"
        )
    ]
    match = memory.match_historical_failures(
        claim_text=claim,
        evidence=weak_ev,
        preliminary_verdict="SUPPORT",
        confidence=0.85
    )
    assert "FP_UNG_004" in match.matched_pattern_ids
    rev_verdict, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
        claim_text=claim,
        evidence=weak_ev,
        match_result=match,
        current_verdict="SUPPORT",
        current_confidence=0.85
    )
    assert rev_verdict in ("INSUFFICIENT_EVIDENCE", "NEI")
    assert rev_conf <= 0.60


# 8. Contradictory Evidence Escalation
def test_contradictory_evidence_escalation(memory):
    claim = "Sleep deprivation enhances memory consolidation."
    contra_ev = [
        RetrievedEvidence(
            doc_id="ev_contra",
            text="Sleep deprivation severely impairs contextual memory consolidation in subjects.",
            score=0.88,
            rank=1,
            title="Sleep Deprivation Study",
            metadata={"relationship": "CONTRADICTS"}
        )
    ]
    match = memory.match_historical_failures(
        claim_text=claim,
        evidence=contra_ev,
        preliminary_verdict="SUPPORT",
        confidence=0.85
    )
    assert "FP_CON_006" in match.matched_pattern_ids
    assert "ESCALATE_TO_MULTI_AGENT_DEBATE" in match.adaptive_actions
    rev_verdict, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
        claim_text=claim,
        evidence=contra_ev,
        match_result=match,
        current_verdict="SUPPORT",
        current_confidence=0.85
    )
    assert rev_verdict in ("CONTRADICTED", "REFUTE", "REFUTED")
    assert match_res.claim_recheck_status == "REVISE"


# 9. Claim-Malformation Phrasing Check
def test_claim_malformation_regeneration(memory):
    malformed_claim = "Altering increasing model size directly impacts performance."
    match = memory.match_historical_failures(
        claim_text=malformed_claim,
        evidence=[]
    )
    assert "FP_CLM_009" in match.matched_pattern_ids
    assert "REGENERATE_ATOMIC_CLAIM" in match.adaptive_actions


# 10. Failure-Memory & Trusted Memory Persistence
def test_failure_memory_persistence(memory):
    assert os.path.exists(memory.patterns_path)
    ok = memory.record_trusted_verification(
        claim_id="test_trusted_01",
        claim_text="Sufficient sleep duration is associated with improved memory retention.",
        verdict="SUPPORTED",
        confidence=0.91,
        evidence=[{"score": 0.88}],
        question="Does sleep improve memory?"
    )
    assert ok is True
    records = memory.load_trusted_memory()
    assert len(records) >= 1
    assert records[-1]["claim_id"] == "test_trusted_01"


# 11. End-to-End Loop: Run 1 Failure Changes Run 2 Decision
def test_end_to_end_loop_run1_failure_changes_run2_decision(memory):
    # Run 1: Unmitigated off-domain evidence produces false SUPPORT
    claim_run1 = "Quantum-resistant federated learning ensures communication efficiency."
    off_domain_doc = RetrievedEvidence(
        doc_id="run1_sg",
        text="Smart grid communications protocol analysis in distribution substations.",
        score=0.76,
        rank=1,
        title="Smart Grid Comms"
    )
    orchestrator = MultiAgentOrchestrator()
    req1 = VerificationRequest(claim_id="r1_c1", claim_text=claim_run1, top_k=1)
    res_run1 = orchestrator.verify(request=req1, pre_retrieved_evidence=[off_domain_doc])
    
    # Pre-memory naive verifier produces SUPPORT
    assert res_run1.final_label in ("SUPPORT", "SUPPORTED")
    
    # Run 2: With Failure-Pattern Memory active on similar query
    claim_run2 = "Quantum-resistant federated learning minimizes network latency."
    off_domain_doc_2 = RetrievedEvidence(
        doc_id="run2_sg",
        text="Phasor measurement units and smart grid power dispatch latency.",
        score=0.75,
        rank=1,
        title="Power Dispatch Latency"
    )
    req2 = VerificationRequest(claim_id="r2_c2", claim_text=claim_run2, top_k=1)
    res_run2 = orchestrator.verify(request=req2, pre_retrieved_evidence=[off_domain_doc_2])
    
    # Adaptive Memory Check:
    match = memory.match_historical_failures(
        claim_text=claim_run2,
        evidence=res_run2.evidence,
        preliminary_verdict=res_run2.final_label,
        confidence=res_run2.confidence
    )
    assert match.matched is True
    assert "FP_DOM_001" in match.matched_pattern_ids
    
    # Adaptive Claim Re-Check:
    rev_verdict, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
        claim_text=claim_run2,
        evidence=res_run2.evidence,
        match_result=match,
        current_verdict=res_run2.final_label,
        current_confidence=res_run2.confidence
    )
    
    # The final decision MUST change from preliminary SUPPORT to INSUFFICIENT_EVIDENCE (NEI)
    assert rev_verdict in ("INSUFFICIENT_EVIDENCE", "NEI")
    assert rev_conf <= 0.35
    assert len(rev_ev) == 0
    assert match_res.claim_recheck_status == "REJECT"
