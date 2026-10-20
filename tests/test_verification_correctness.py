"""Comprehensive regression test suite for CiteGuard Verification Correctness.

Validates the 12 critical scientific verification invariants:
1. High retrieval relevance without entailment does NOT produce SUPPORTED.
2. Universal claims ("all", "always") with narrow evidence trigger F3 (Overgeneralization).
3. Contradictory findings in evidence trigger F7 (Contradictory Evidence) -> CONTRADICTED.
4. Causal claims on correlational evidence trigger F4 (Causal Leap) -> PARTIALLY_SUPPORTED.
5. Domain/context mismatch triggers F6 (Population/Context Mismatch) -> INSUFFICIENT_EVIDENCE.
6. Outdated evidence triggers F5 (Outdated Evidence) -> PARTIALLY_SUPPORTED.
7. Directional inversion triggers F8 / F7 -> CONTRADICTED / REFUTE.
8. Citation mismatch / unverified DOI triggers F2 (Citation Mismatch) -> INSUFFICIENT_EVIDENCE.
9. Clean supported claim produces SUPPORTED and is recorded in trusted memory.
10. Failure-pattern memory matching triggers Claim Re-Check -> REVISE/REJECT.
11. API error / empty retrieval handling outputs calibrated NEI (confidence <= 0.35).
12. UI / API response verdicts strictly match backend report normalized verdicts.
"""

import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient

from src.verification.schema import (
    VerificationRequest,
    RetrievedEvidence,
    VerificationResult,
    AgentDecision
)
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.skeptic_agent import SkepticAgent, SkepticResult
from src.evaluation.failure_memory import FailureMemory, PatternMatchResult
from src.evaluation.failure_taxonomy import CanonicalFailureCode
from src.reporting.generator import ReportGenerator, normalize_verdict
from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.ui.server import app


# ----------------------------------------------------------------------------
# 1. High Relevance Without Entailment
# ----------------------------------------------------------------------------
def test_scenario_01_relevance_without_entailment_not_supported():
    """Retrieval relevance is not claim support. A passage sharing topic keywords
    without asserting the claim proposition must not produce a SUPPORTED verdict.
    """
    claim = "CRISPR base editing causes telomere lengthening in cardiomyocytes."
    # Top-retrieved document by search on cardiomyocyte topic keywords (score 0.91),
    # but zero entailment regarding CRISPR base editing or telomere lengthening
    evidence = [
        RetrievedEvidence(
            doc_id="crispr_cardio_01",
            text="Cardiomyocyte cellular differentiation protocols using human pluripotent stem cell cultures.",
            score=0.91,
            rank=1,
            title="Heart Cell Differentiation"
        )
    ]
    orchestrator = MultiAgentOrchestrator()
    req = VerificationRequest(claim_id="scen_01", claim_text=claim, top_k=1)
    res = orchestrator.verify(request=req, pre_retrieved_evidence=evidence)

    # Invariant: Must NOT conclude SUPPORTED
    norm_v = normalize_verdict(res.final_label)
    assert norm_v in ("INSUFFICIENT_EVIDENCE", "PARTIALLY_SUPPORTED")
    assert norm_v != "SUPPORTED"


# ----------------------------------------------------------------------------
# 2. Universal Quantifier With Narrow Evidence (F3 Overgeneralization)
# ----------------------------------------------------------------------------
def test_scenario_02_universal_claim_triggers_f3_overgeneralization():
    """Universal assertions ('all', 'always', 'in all networks') on narrow evidence
    must be challenged by SkepticAgent as F3 (Overgeneralization) and capped in confidence.
    """
    claim = "All deep neural networks always achieve zero generalization error in every domain."
    evidence = [
        RetrievedEvidence(
            doc_id="nn_sample_01",
            text="A convolutional neural network achieved high validation accuracy on CIFAR-10 image classification.",
            score=0.78,
            rank=1,
            title="Image Classification Evaluation"
        )
    ]
    skeptic = SkepticAgent()
    sk_res = skeptic.evaluate(claim_text=claim, evidence=evidence)

    assert sk_res.challenged is True
    assert CanonicalFailureCode.F3 in sk_res.failure_candidates or CanonicalFailureCode.F3 == sk_res.primary_failure_code
    assert any("F3" in obj or "Overgeneralization" in obj or "Universal" in obj for obj in sk_res.objections)


# ----------------------------------------------------------------------------
# 3. Contradictory Evidence (F7 Contradictory Evidence)
# ----------------------------------------------------------------------------
def test_scenario_03_contradictory_evidence_triggers_f7_refutation():
    """When candidate evidence contradicts the claim statement, Skeptic and Verifier
    must identify F7 and resolve to CONTRADICTED / REFUTE.
    """
    claim = "Short sleep duration increases cognitive performance in adults."
    evidence = [
        RetrievedEvidence(
            doc_id="sleep_ref_01",
            text="Short sleep duration does not increase cognitive performance and causes sustained attention deficits in adults.",
            score=0.88,
            rank=1,
            title="Sleep Restriction Study"
        )
    ]
    orchestrator = MultiAgentOrchestrator()
    req = VerificationRequest(claim_id="scen_03", claim_text=claim, top_k=1)
    res = orchestrator.verify(request=req, pre_retrieved_evidence=evidence)

    norm_v = normalize_verdict(res.final_label)
    assert norm_v == "CONTRADICTED"


# ----------------------------------------------------------------------------
# 4. Correlation as Causation (F4 Causal Leap)
# ----------------------------------------------------------------------------
def test_scenario_04_causal_claim_on_correlational_evidence_triggers_f4():
    """Asserting causal mechanisms ('causes', 'leads directly to') on observational
    data triggers F4 Causal Leap and adapts to PARTIALLY_SUPPORTED.
    """
    claim = "Coffee consumption causes a reduction in cardiovascular disease mortality."
    evidence = [
        RetrievedEvidence(
            doc_id="coffee_obs_01",
            text="Habitual coffee consumption is correlated with lower cardiovascular mortality in an observational cohort study.",
            score=0.85,
            rank=1,
            title="Cohort Dietary Patterns"
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        patterns_path = os.path.join(tmp_dir, "patterns.json")
        trusted_path = os.path.join(tmp_dir, "trusted.json")
        memory = FailureMemory(patterns_path=patterns_path, trusted_memory_path=trusted_path)

        match = memory.match_historical_failures(
            claim_text=claim,
            evidence=evidence,
            preliminary_verdict="SUPPORT",
            confidence=0.85
        )

        assert match.matched is True
        assert match.canonical_failure_code == "F4"

        rev_v, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
            claim_text=claim,
            evidence=evidence,
            match_result=match,
            current_verdict="SUPPORT",
            current_confidence=0.85
        )

        assert normalize_verdict(rev_v) == "PARTIALLY_SUPPORTED"
        assert rev_conf <= 0.60
        assert match_res.claim_recheck_status == "REVISE"


# ----------------------------------------------------------------------------
# 5. Domain / Context Mismatch (F6 Population/Context Mismatch)
# ----------------------------------------------------------------------------
def test_scenario_05_domain_mismatch_triggers_f6_insufficient():
    """Off-domain evidence (e.g. smart grid literature for quantum FL) triggers F6
    and is rejected by Claim Re-Check to INSUFFICIENT_EVIDENCE.
    """
    claim = "Quantum-resistant federated learning prevents eavesdropping."
    evidence = [
        RetrievedEvidence(
            doc_id="off_grid_01",
            text="Smart grid communications protocol analysis in distribution substations.",
            score=0.76,
            rank=1,
            title="Smart Grid Comms"
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        patterns_path = os.path.join(tmp_dir, "patterns.json")
        trusted_path = os.path.join(tmp_dir, "trusted.json")
        memory = FailureMemory(patterns_path=patterns_path, trusted_memory_path=trusted_path)

        match = memory.match_historical_failures(
            claim_text=claim,
            evidence=evidence,
            question="Quantum federated learning security",
            preliminary_verdict="SUPPORT",
            confidence=0.80
        )

        assert match.matched is True
        assert match.canonical_failure_code == "F6"

        rev_v, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
            claim_text=claim,
            evidence=evidence,
            match_result=match,
            current_verdict="SUPPORT",
            current_confidence=0.80,
            question="Quantum federated learning security"
        )

        assert normalize_verdict(rev_v) == "INSUFFICIENT_EVIDENCE"
        assert len(rev_ev) == 0
        assert match_res.claim_recheck_status == "REJECT"


# ----------------------------------------------------------------------------
# 6. Outdated Evidence (F5 Outdated Evidence)
# ----------------------------------------------------------------------------
def test_scenario_06_outdated_evidence_triggers_f5():
    """Evidence marked as superseded or chronologically obsolete triggers F5
    and adapts verdict to PARTIALLY_SUPPORTED.
    """
    claim = "Neural language models scale cross-entropy loss predictably."
    evidence = [
        RetrievedEvidence(
            doc_id="old_paper_01",
            text="N-gram backoff models predict cross-entropy loss on small corpora.",
            score=0.72,
            rank=1,
            title="Historical Language Modeling",
            metadata={"is_superseded": True, "publication_year": 1995}
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        patterns_path = os.path.join(tmp_dir, "patterns.json")
        trusted_path = os.path.join(tmp_dir, "trusted.json")
        memory = FailureMemory(patterns_path=patterns_path, trusted_memory_path=trusted_path)

        match = memory.match_historical_failures(
            claim_text=claim,
            evidence=evidence,
            preliminary_verdict="SUPPORT",
            confidence=0.80
        )

        assert match.matched is True
        assert match.canonical_failure_code == "F5"

        rev_v, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
            claim_text=claim,
            evidence=evidence,
            match_result=match,
            current_verdict="SUPPORT",
            current_confidence=0.80
        )

        assert normalize_verdict(rev_v) == "PARTIALLY_SUPPORTED"
        assert rev_conf <= 0.55


# ----------------------------------------------------------------------------
# 7. Directional Inversion / Misinterpretation (F8)
# ----------------------------------------------------------------------------
def test_scenario_07_directional_inversion_triggers_refute():
    """When evidence indicates the inverse direction of the asserted effect,
    verification must classify relationship as CONTRADICTS and reject the claim.
    """
    claim = "High sodium intake reduces arterial blood pressure."
    evidence = [
        RetrievedEvidence(
            doc_id="sodium_bp_01",
            text="High sodium intake increases arterial blood pressure and cardiovascular risk in hypertensive patients.",
            score=0.90,
            rank=1,
            title="Sodium and Hypertension"
        )
    ]
    orchestrator = MultiAgentOrchestrator()
    req = VerificationRequest(claim_id="scen_07", claim_text=claim, top_k=1)
    res = orchestrator.verify(request=req, pre_retrieved_evidence=evidence)

    norm_v = normalize_verdict(res.final_label)
    assert norm_v == "CONTRADICTED"


# ----------------------------------------------------------------------------
# 8. Citation / Provenance Mismatch (F2 Citation Mismatch)
# ----------------------------------------------------------------------------
def test_scenario_08_citation_mismatch_triggers_f2_insufficient():
    """Unverified bibliographic provenance or missing registry DOIs trigger F2
    and result in INSUFFICIENT_EVIDENCE.
    """
    claim = "Literature findings indexed under unverified registry DOI."
    evidence = [
        RetrievedEvidence(
            doc_id="unverified_doc_01",
            text="Some scientific assertion without verified publication metadata.",
            score=0.75,
            rank=1,
            title="Unindexed Preprint",
            metadata={"citation_mismatch": True, "metadata_unverified": True}
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        patterns_path = os.path.join(tmp_dir, "patterns.json")
        trusted_path = os.path.join(tmp_dir, "trusted.json")
        memory = FailureMemory(patterns_path=patterns_path, trusted_memory_path=trusted_path)

        match = memory.match_historical_failures(
            claim_text=claim,
            evidence=evidence,
            preliminary_verdict="SUPPORT",
            confidence=0.75
        )

        assert match.matched is True
        assert match.canonical_failure_code == "F2"

        rev_v, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
            claim_text=claim,
            evidence=evidence,
            match_result=match,
            current_verdict="SUPPORT",
            current_confidence=0.75
        )

        assert normalize_verdict(rev_v) == "INSUFFICIENT_EVIDENCE"
        assert match_res.claim_recheck_status == "REJECT"


# ----------------------------------------------------------------------------
# 9. Clean Supported Claim -> Memory Gate -> Trusted Memory
# ----------------------------------------------------------------------------
def test_scenario_09_clean_supported_claim_admitted_to_trusted_memory():
    """A claim with strong exact evidence, high confidence, zero contradictions,
    and zero failure matches passes the memory gate and is admitted to trusted memory.
    """
    claim = "Sleep deprivation impairs spatial memory consolidation in rodents."
    evidence = [
        RetrievedEvidence(
            doc_id="sleep_rodent_01",
            text="Sleep deprivation impairs spatial memory consolidation and hippocampal LTP in rodents.",
            score=0.92,
            rank=1,
            title="Rodent Memory Consolidation"
        )
    ]
    orchestrator = MultiAgentOrchestrator()
    req = VerificationRequest(claim_id="scen_09", claim_text=claim, top_k=1)
    res = orchestrator.verify(request=req, pre_retrieved_evidence=evidence)

    assert normalize_verdict(res.final_label) == "SUPPORTED"
    assert res.confidence >= 0.70

    with tempfile.TemporaryDirectory() as tmp_dir:
        trusted_path = os.path.join(tmp_dir, "trusted.json")
        memory = FailureMemory(trusted_memory_path=trusted_path)

        recorded = memory.record_trusted_verification(
            claim_id="scen_09",
            claim_text=claim,
            verdict=res.final_label,
            confidence=res.confidence,
            evidence=evidence,
            question="Sleep and Memory"
        )

        assert recorded is True
        records = memory.load_trusted_memory()
        assert len(records) == 1
        assert records[0]["claim_id"] == "scen_09"


# ----------------------------------------------------------------------------
# 10. Failure-Pattern Memory Match Triggers Re-Check
# ----------------------------------------------------------------------------
def test_scenario_10_failure_memory_match_executes_claim_recheck():
    """When historical failure pattern triggers (e.g. FP_UNG_004 subthreshold support),
    Claim Re-Check adapts verdict and records prevention flag.
    """
    claim = "Weak assertion supported by sub-threshold evidence."
    evidence = [
        RetrievedEvidence(
            doc_id="sub_01",
            text="Peripheral contextual sentence.",
            score=0.55,
            rank=1,
            title="Peripheral Study"
        )
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        patterns_path = os.path.join(tmp_dir, "patterns.json")
        trusted_path = os.path.join(tmp_dir, "trusted.json")
        memory = FailureMemory(patterns_path=patterns_path, trusted_memory_path=trusted_path)

        match = memory.match_historical_failures(
            claim_text=claim,
            evidence=evidence,
            preliminary_verdict="SUPPORT",
            confidence=0.75
        )

        assert match.matched is True
        assert "FP_UNG_004" in match.matched_pattern_ids

        rev_v, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
            claim_text=claim,
            evidence=evidence,
            match_result=match,
            current_verdict="SUPPORT",
            current_confidence=0.75
        )

        assert normalize_verdict(rev_v) == "INSUFFICIENT_EVIDENCE"
        assert match_res.prevention_triggered is True
        assert match_res.claim_recheck_status == "REJECT"


# ----------------------------------------------------------------------------
# 11. Empty Retrieval / API Failure Handling
# ----------------------------------------------------------------------------
def test_scenario_11_empty_retrieval_calibrated_nei():
    """Empty retrieval returns calibrated INSUFFICIENT_EVIDENCE (confidence <= 0.35)
    and does not crash or synthesize artificial support.
    """
    claim = "Obscure unstudied biochemical assertion."
    orchestrator = MultiAgentOrchestrator()
    req = VerificationRequest(claim_id="scen_11", claim_text=claim, top_k=5)
    res = orchestrator.verify(request=req, pre_retrieved_evidence=[])

    assert normalize_verdict(res.final_label) == "INSUFFICIENT_EVIDENCE"
    assert res.confidence <= 0.35
    assert len(res.evidence) == 0
    assert res.resolution_method == "insufficient_evidence_fallback"


# ----------------------------------------------------------------------------
# 12. UI / API Verdict Normalization Agreement
# ----------------------------------------------------------------------------
def test_scenario_12_ui_and_backend_verdict_agreement():
    """Verdicts returned by the API server and report generator must match
    the canonical taxonomy: SUPPORTED, PARTIALLY_SUPPORTED, INSUFFICIENT_EVIDENCE, CONTRADICTED.
    """
    client = TestClient(app)
    resp = client.post(
        "/api/verify",
        json={
            "question": "How does sleep loss impair spatial memory retention in mammals?",
            "mode": "openalex_hybrid",
            "top_k": 3,
            "provider": "mock"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "claims" in data
    assert len(data["claims"]) > 0
    first_verdict = data["claims"][0]["verdict"]
    assert first_verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONTRADICTED")

    # Verify ReportGenerator uses identical canonical vocabulary
    with tempfile.TemporaryDirectory() as tmp_dir:
        rg = ReportGenerator(output_dir=tmp_dir)
        rq = ResearchQuestion(
            question_id="q_sync",
            research_question="Does sleep loss impair spatial memory?",
            generated_claims=[]
        )
        v_res = [
            VerificationResult(claim_id="c1", claim_text="Claim 1", final_label=first_verdict, confidence=0.85)
        ]
        files = rg.generate_report(rq, v_res)
        with open(files["json_path"], "r", encoding="utf-8") as f:
            report_data = json.load(f)

        assert report_data["claims"][0]["verdict"] == first_verdict
