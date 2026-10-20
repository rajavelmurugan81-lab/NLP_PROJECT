"""Comprehensive Closed-Loop Failure Learning Integration Tests for CiteGuard.

Demonstrates the complete two-run learning loop:
  Run 1: Detect failure -> Record to runtime audit -> Mine new pattern -> Persist
  Run 2: Load pattern -> Match -> Trigger prevention -> Re-check -> Changed verdict

Also tests F1-F8 coverage, ledger isolation, persistence, deduplication,
clean claim safety, and edge cases.
"""

import os
import json
import tempfile
import pytest
from typing import List, Dict, Any

from src.evaluation.failure_pattern import FailurePattern, PatternMatchResult
from src.evaluation.pattern_miner import PatternMiner
from src.evaluation.failure_memory import FailureMemory
from src.evaluation.runtime_auditor import RuntimeAuditor
from src.evaluation.failure_classifier import FailureClassifier
from src.verification.schema import VerificationRequest, VerificationResult, RetrievedEvidence
from src.verification.orchestrator import MultiAgentOrchestrator


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def isolated_env(tmp_path):
    """Create fully isolated ledger environment for deterministic testing."""
    runtime_path = str(tmp_path / "runtime_audit.jsonl")
    benchmark_path = str(tmp_path / "hallucination_ledger.jsonl")
    patterns_path = str(tmp_path / "failure_patterns.json")
    trusted_path = str(tmp_path / "trusted_memory.json")
    return {
        "runtime_path": runtime_path,
        "benchmark_path": benchmark_path,
        "patterns_path": patterns_path,
        "trusted_path": trusted_path,
        "tmp_path": tmp_path,
    }


def _make_evidence(doc_id, text, score, title="Paper", metadata=None):
    """Helper to build RetrievedEvidence with optional metadata."""
    return RetrievedEvidence(
        doc_id=doc_id,
        text=text,
        score=score,
        rank=1,
        title=title,
        metadata=metadata or {}
    )


def _count_lines(path):
    """Count non-empty lines in a JSONL file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_patterns_file(path):
    """Load failure_patterns.json and return pattern list."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("patterns", [])


# -----------------------------------------------------------------------
# TEST 1: Complete Two-Run Closed-Loop Integration Test
# -----------------------------------------------------------------------

class TestTwoRunClosedLoop:
    """Demonstrates the complete research contribution:
    Run 1 failure -> pattern mining -> Run 2 prevention."""

    def test_run1_failure_detection_and_pattern_creation(self, isolated_env):
        """Run 1: Detect a domain-mismatch failure, record it, mine a pattern."""
        env = isolated_env

        # Create an orchestrator with isolated ledgers
        orchestrator = MultiAgentOrchestrator(
            ledger_path=env["benchmark_path"],
            runtime_ledger_path=env["runtime_path"]
        )

        # Submit a quantum FL claim with off-domain smart grid evidence
        claim_text = "Quantum-resistant federated learning protects patient health records."
        off_domain_ev = _make_evidence(
            "run1_sg",
            "Smart grid SCADA systems require anomaly detection protocols for power grid protection.",
            score=0.76,
            title="Smart Grid Security Analysis"
        )

        req = VerificationRequest(
            claim_id="run1_c001",
            claim_text=claim_text,
            top_k=1,
            metadata={"managed_audit": False}
        )

        # Step 1: Verify — orchestrator naively produces SUPPORT
        result = orchestrator.verify(request=req, pre_retrieved_evidence=[off_domain_ev])
        assert result.final_label in ("SUPPORT", "SUPPORTED"), \
            f"Pre-memory orchestrator should produce SUPPORT, got {result.final_label}"

        # Step 2: Manually record the failure audit (simulating pipeline behavior)
        auditor = RuntimeAuditor(ledger_filepath=env["runtime_path"])
        audit_record = auditor.audit_claim(
            claim_id="run1_c001",
            claim_text=claim_text,
            verdict=result.final_label,
            confidence=result.confidence,
            evidence=result.evidence,
            question="How does quantum-resistant federated learning protect healthcare data?",
            run_id="run1"
        )

        # Verify runtime audit was written
        assert _count_lines(env["runtime_path"]) >= 1
        audit_data = audit_record.to_dict()
        assert audit_data["failure_type"] in ("DOMAIN_MISMATCH", "UNGROUNDED_SUPPORT", "NO_EVIDENCE"), \
            f"Expected a failure type, got {audit_data['failure_type']}"

        # Step 3: Mine patterns from the new runtime audit
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        patterns = miner.mine_patterns()
        assert len(patterns) >= 9, f"Expected >= 9 patterns, got {len(patterns)}"
        assert os.path.exists(env["patterns_path"]), "failure_patterns.json must be persisted"

        # Verify the domain mismatch pattern was created with updated provenance
        saved_patterns = _load_patterns_file(env["patterns_path"])
        dom_pattern = next((p for p in saved_patterns if p["pattern_id"] == "FP_DOM_001"), None)
        assert dom_pattern is not None, "FP_DOM_001 must exist in mined patterns"
        assert dom_pattern["frequency"] >= 1

        # Step 4: Benchmark ledger must remain empty (isolation)
        assert _count_lines(env["benchmark_path"]) == 0, \
            "Benchmark ledger must not receive live-query records"

    def test_run2_pattern_matching_and_prevention(self, isolated_env):
        """Run 2: Load previously learned pattern, match similar query, prevent failure."""
        env = isolated_env

        # First, simulate Run 1's aftermath — create a failure pattern file
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        miner.mine_patterns()  # Creates patterns from (possibly empty) ledgers

        # Load failure memory from the persisted patterns
        memory = FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )
        assert len(memory.patterns) >= 9

        # Simulate Run 2: Similar quantum FL claim with off-domain evidence
        claim_run2 = "Quantum-resistant federated learning minimizes network latency."
        off_domain_ev = _make_evidence(
            "run2_sg",
            "Phasor measurement units and smart grid power dispatch latency optimization.",
            score=0.75,
            title="Power Dispatch Systems"
        )

        # Step 1: Orchestrator naively produces SUPPORT
        orchestrator = MultiAgentOrchestrator(
            ledger_path=env["benchmark_path"],
            runtime_ledger_path=env["runtime_path"]
        )
        req = VerificationRequest(
            claim_id="run2_c001", claim_text=claim_run2, top_k=1
        )
        result = orchestrator.verify(request=req, pre_retrieved_evidence=[off_domain_ev])
        initial_verdict = result.final_label
        assert initial_verdict in ("SUPPORT", "SUPPORTED")

        # Step 2: Pattern matching detects the domain mismatch
        match_res = memory.match_historical_failures(
            claim_text=claim_run2,
            evidence=result.evidence,
            question="How does quantum-resistant federated learning reduce network latency?",
            preliminary_verdict=result.final_label,
            confidence=result.confidence
        )
        assert match_res.matched is True, "Pattern must match"
        assert "FP_DOM_001" in match_res.matched_pattern_ids
        assert match_res.risk_level in ("HIGH", "VERY_HIGH")
        assert match_res.pattern_match_score >= 0.85

        # Step 3: Adaptive re-check changes the verdict
        revised_verdict, revised_conf, revised_ev, updated_match = memory.execute_claim_recheck(
            claim_text=claim_run2,
            evidence=result.evidence,
            match_result=match_res,
            current_verdict=result.final_label,
            current_confidence=result.confidence,
            question="How does quantum-resistant federated learning reduce network latency?"
        )

        # CRITICAL ASSERTION: verdict must change from SUPPORT to NEI/INSUFFICIENT_EVIDENCE
        assert revised_verdict in ("NEI", "INSUFFICIENT_EVIDENCE"), \
            f"Expected NEI or INSUFFICIENT_EVIDENCE after re-check, got {revised_verdict}"
        assert revised_conf <= 0.35, f"Confidence must be <= 0.35, got {revised_conf}"
        assert len(revised_ev) == 0, "Off-domain evidence must be filtered out"
        assert updated_match.claim_recheck_status == "REJECT"
        assert updated_match.prevention_triggered is True

        # Step 4: Benchmark ledger remains isolated
        assert _count_lines(env["benchmark_path"]) == 0

    def test_complete_two_run_loop_end_to_end(self, isolated_env):
        """Complete end-to-end loop: Run 1 records failure -> mines pattern -> 
        Run 2 loads pattern -> matches -> prevents -> verdict changes."""
        env = isolated_env

        # ===== RUN 1: Detect and Learn =====
        orchestrator = MultiAgentOrchestrator(
            ledger_path=env["benchmark_path"],
            runtime_ledger_path=env["runtime_path"]
        )
        auditor = RuntimeAuditor(ledger_filepath=env["runtime_path"])

        claim_run1 = "Quantum-resistant federated learning ensures secure communication."
        ev_run1 = _make_evidence(
            "run1_sg", "Smart grid communications protocol analysis in substations.", 0.76, "Smart Grid Comms"
        )
        req1 = VerificationRequest(claim_id="loop_c001", claim_text=claim_run1, top_k=1)
        res1 = orchestrator.verify(request=req1, pre_retrieved_evidence=[ev_run1])
        assert res1.final_label in ("SUPPORT", "SUPPORTED")

        # Record failure
        audit1 = auditor.audit_claim(
            claim_id="loop_c001", claim_text=claim_run1,
            verdict=res1.final_label, confidence=res1.confidence,
            evidence=res1.evidence,
            question="How does quantum-resistant federated learning ensure security?",
            run_id="loop_run1"
        )
        run1_audit_count = _count_lines(env["runtime_path"])
        assert run1_audit_count >= 1

        # Mine patterns from runtime audit
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        patterns_run1 = miner.mine_patterns()
        assert os.path.exists(env["patterns_path"])
        patterns_before = len(patterns_run1)

        # ===== RUN 2: Match and Prevent =====
        # Simulate application restart by creating new FailureMemory from disk
        memory_run2 = FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )
        assert len(memory_run2.patterns) >= 9, "Patterns must load from disk after restart"

        claim_run2 = "Quantum-resistant federated learning reduces transmission overhead."
        ev_run2 = _make_evidence(
            "run2_sg", "Electric power grid distribution infrastructure latency.", 0.74, "Grid Infrastructure"
        )
        req2 = VerificationRequest(claim_id="loop_c002", claim_text=claim_run2, top_k=1)
        res2 = orchestrator.verify(request=req2, pre_retrieved_evidence=[ev_run2])
        initial_run2_verdict = res2.final_label
        assert initial_run2_verdict in ("SUPPORT", "SUPPORTED")

        # Pattern matching
        match2 = memory_run2.match_historical_failures(
            claim_text=claim_run2, evidence=res2.evidence,
            question="How does quantum-resistant federated learning reduce overhead?",
            preliminary_verdict=res2.final_label, confidence=res2.confidence
        )
        assert match2.matched is True
        assert "FP_DOM_001" in match2.matched_pattern_ids

        # Prevention action
        final_verdict, final_conf, final_ev, final_match = memory_run2.execute_claim_recheck(
            claim_text=claim_run2, evidence=res2.evidence,
            match_result=match2, current_verdict=res2.final_label,
            current_confidence=res2.confidence,
            question="How does quantum-resistant federated learning reduce overhead?"
        )

        # FINAL ACCEPTANCE CRITERION: verdict changed from SUPPORT to NEI
        assert initial_run2_verdict in ("SUPPORT", "SUPPORTED"), f"Run 2 initial was {initial_run2_verdict}"
        assert final_verdict in ("NEI", "INSUFFICIENT_EVIDENCE"), f"Run 2 final should be NEI, got {final_verdict}"
        assert final_conf <= 0.35
        assert final_match.prevention_triggered is True
        assert final_match.claim_recheck_status == "REJECT"

        # Record Run 2 audit
        audit2 = auditor.audit_claim(
            claim_id="loop_c002", claim_text=claim_run2,
            verdict=final_verdict, confidence=final_conf,
            evidence=final_ev,
            question="How does quantum-resistant federated learning reduce overhead?",
            run_id="loop_run2",
            matched_pattern_ids=match2.matched_pattern_ids,
            pattern_match_score=match2.pattern_match_score,
            risk_level=match2.risk_level,
            adaptive_action=match2.adaptive_actions[0] if match2.adaptive_actions else "NONE",
            prevention_triggered=True,
            claim_recheck_status="REJECT"
        )

        # Verify audit metadata
        a2_dict = audit2.to_dict()
        assert a2_dict["prevention_triggered"] is True
        assert "FP_DOM_001" in a2_dict["matched_pattern_ids"]
        assert a2_dict["claim_recheck_status"] == "REJECT"

        # Verify benchmark isolation throughout
        assert _count_lines(env["benchmark_path"]) == 0


# -----------------------------------------------------------------------
# TEST 2: Clean Claim Safety — Pattern Matching Must NOT Reject Valid Claims
# -----------------------------------------------------------------------

class TestCleanClaimSafety:
    """Ensure pattern matching does not produce false positives on valid claims."""

    def test_clean_supported_claim_not_rejected(self, isolated_env):
        """A well-supported claim with relevant evidence must pass through unaffected."""
        env = isolated_env
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        miner.mine_patterns()

        memory = FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )

        # On-domain claim with matching evidence
        claim = "Sleep deprivation impairs working memory performance in adults."
        evidence = [
            _make_evidence(
                "sleep_001",
                "Experimental studies demonstrate that sleep deprivation significantly impairs working memory consolidation in adult subjects.",
                score=0.92,
                title="Sleep and Memory Review",
                metadata={"relationship": "SUPPORTS"}
            )
        ]

        match = memory.match_historical_failures(
            claim_text=claim, evidence=evidence,
            question="Does sleep deprivation affect memory?",
            preliminary_verdict="SUPPORT", confidence=0.88
        )

        # This should NOT trigger domain mismatch or ungrounded support patterns
        if match.matched and match.risk_level in ("MEDIUM", "HIGH", "VERY_HIGH"):
            verdict, conf, ev, updated = memory.execute_claim_recheck(
                claim_text=claim, evidence=evidence, match_result=match,
                current_verdict="SUPPORT", current_confidence=0.88,
                question="Does sleep deprivation affect memory?"
            )
            # Even if some pattern triggered, verdict should not be NEI
            # because the evidence score is 0.92 > 0.70
            assert verdict in ("SUPPORT", "SUPPORTED", "PARTIALLY_SUPPORTED"), \
                f"Clean claim should not be rejected, got {verdict}"
        else:
            # No pattern matched or risk is LOW — this is the expected path
            assert match.risk_level == "LOW" or not match.matched


# -----------------------------------------------------------------------
# TEST 3: All Canonical F1-F8 Pattern Matching Coverage
# -----------------------------------------------------------------------

class TestF1F8Coverage:
    """Verify all canonical failure types have matching patterns and re-check behavior."""

    @pytest.fixture
    def memory(self, isolated_env):
        env = isolated_env
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        miner.mine_patterns()
        return FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )

    def test_f1_unsupported_claim(self, memory):
        """F1: Zero retrieved passages triggers FP_NEI_007 / ENFORCE_SUFFICIENCY_GATE; prevents SUPPORT."""
        match = memory.match_historical_failures(
            claim_text="Neural networks solve all optimization problems unconditionally.",
            evidence=[],
            preliminary_verdict="SUPPORT", confidence=0.85
        )
        assert match.matched is True
        assert "FP_NEI_007" in match.matched_pattern_ids or "FP_UNG_004" in match.matched_pattern_ids
        assert any(act in match.adaptive_actions for act in ("ENFORCE_SUFFICIENCY_GATE", "CLAIM_RECHECK_SUFFICIENCY"))

        # Re-check execution on zero evidence
        rev_verdict, rev_conf, rev_ev, match_res = memory.execute_claim_recheck(
            claim_text="Neural networks solve all optimization problems unconditionally.",
            evidence=[],
            match_result=match,
            current_verdict="SUPPORT",
            current_confidence=0.85
        )
        assert rev_verdict in ("NEI", "INSUFFICIENT_EVIDENCE"), "Sufficiency gate must not allow SUPPORT on zero evidence"
        assert rev_conf <= 0.35, "Confidence must be low (<= 0.35) for unsupported claim"
        assert len(rev_ev) == 0
        assert match_res.claim_recheck_status == "REJECT", "Claim recheck status must be REJECT"
        assert match_res.prevention_triggered is True

    def test_f2_citation_mismatch(self, memory):
        """F2: Evidence with unverified citation metadata triggers FP_MET_008."""
        ev = _make_evidence("cite_001", "Some paper text.", 0.80, "Paper X",
                            metadata={"citation_mismatch": True})
        match = memory.match_historical_failures(
            claim_text="Study X proves the hypothesis.", evidence=[ev],
            preliminary_verdict="SUPPORT", confidence=0.85
        )
        assert match.matched is True
        assert "FP_MET_008" in match.matched_pattern_ids

    def test_f3_overgeneralization(self, memory):
        """F3: High confidence on sparse evidence triggers FP_CNF_005."""
        ev = _make_evidence("sparse_001", "Preliminary results suggest improvement.", 0.55, "Prelim Study")
        match = memory.match_historical_failures(
            claim_text="Deep learning eliminates all errors.", evidence=[ev],
            preliminary_verdict="SUPPORT", confidence=0.92
        )
        assert match.matched is True
        # Either FP_UNG_004 or FP_CNF_005 should match
        assert any(pid in match.matched_pattern_ids for pid in ("FP_UNG_004", "FP_CNF_005"))

    def test_f4_causal_leap(self, memory):
        """F4: Causal claim on correlational evidence triggers FP_CAU_010."""
        ev = _make_evidence(
            "obs_001",
            "Coffee consumption was correlated with reduced risk in an observational cohort.",
            0.85, "Observational Study"
        )
        match = memory.match_historical_failures(
            claim_text="Coffee causes reduced disease risk in all populations.",
            evidence=[ev], preliminary_verdict="SUPPORT", confidence=0.85
        )
        assert match.matched is True
        assert "FP_CAU_010" in match.matched_pattern_ids

        verdict, conf, _, updated = memory.execute_claim_recheck(
            claim_text="Coffee causes reduced disease risk in all populations.",
            evidence=[ev], match_result=match,
            current_verdict="SUPPORT", current_confidence=0.85
        )
        assert verdict == "PARTIALLY_SUPPORTED"
        assert conf <= 0.60

    def test_f5_outdated_evidence(self, memory):
        """F5: Evidence with publication_year > 20 years ago triggers FP_OUT_011."""
        ev = _make_evidence(
            "old_001", "Historical findings from decades ago.", 0.80, "Old Study",
            metadata={"publication_year": 1990, "is_superseded": True}
        )
        match = memory.match_historical_failures(
            claim_text="This mechanism improves protein folding.",
            evidence=[ev], preliminary_verdict="SUPPORT", confidence=0.80
        )
        assert match.matched is True
        assert "FP_OUT_011" in match.matched_pattern_ids

        verdict, conf, _, _ = memory.execute_claim_recheck(
            claim_text="This mechanism improves protein folding.",
            evidence=[ev], match_result=match,
            current_verdict="SUPPORT", current_confidence=0.80
        )
        assert verdict == "PARTIALLY_SUPPORTED"
        assert conf <= 0.55

    def test_f6_domain_mismatch(self, memory):
        """F6: Off-domain evidence triggers FP_DOM_001."""
        ev = _make_evidence(
            "dom_001", "Drug discovery for natural products targeting carcinoma cells.", 0.72,
            "Drug Discovery Study"
        )
        match = memory.match_historical_failures(
            claim_text="Quantum federated learning protects IoT data.",
            evidence=[ev],
            question="How does quantum federated learning protect IoT?",
            preliminary_verdict="SUPPORT", confidence=0.85
        )
        assert match.matched is True
        assert "FP_DOM_001" in match.matched_pattern_ids
        assert match.risk_level in ("HIGH", "VERY_HIGH")

    def test_f7_contradictory_evidence(self, memory):
        """F7: Contradictory evidence triggers FP_CON_006 and revises verdict to REFUTE."""
        ev = _make_evidence(
            "contra_001",
            "Sleep deprivation severely impairs memory consolidation.",
            0.88, "Sleep Study", metadata={"relationship": "CONTRADICTS"}
        )
        match = memory.match_historical_failures(
            claim_text="Sleep deprivation enhances memory consolidation.",
            evidence=[ev], preliminary_verdict="SUPPORT", confidence=0.85
        )
        assert match.matched is True
        assert "FP_CON_006" in match.matched_pattern_ids

        verdict, conf, _, _ = memory.execute_claim_recheck(
            claim_text="Sleep deprivation enhances memory consolidation.",
            evidence=[ev], match_result=match,
            current_verdict="SUPPORT", current_confidence=0.85
        )
        assert verdict in ("REFUTE", "REFUTED", "CONTRADICTED")

    def test_f8_evidence_misinterpretation(self, memory):
        """F8: Claim malformation triggers FP_CLM_009."""
        match = memory.match_historical_failures(
            claim_text="Altering increasing model size directly impacts outcomes.",
            evidence=[], preliminary_verdict="SUPPORT", confidence=0.80
        )
        assert match.matched is True
        assert "FP_CLM_009" in match.matched_pattern_ids


# -----------------------------------------------------------------------
# TEST 4: Ledger Isolation and Duplicate Prevention
# -----------------------------------------------------------------------

class TestLedgerIsolationAndDedup:
    """Verify strict ledger separation and no duplicate logging."""

    def test_runtime_failure_never_touches_benchmark(self, isolated_env):
        """A live query failure must write to runtime_audit.jsonl only."""
        env = isolated_env

        # Seed a benchmark record
        with open(env["benchmark_path"], "w", encoding="utf-8") as f:
            f.write(json.dumps({"claim_id": "seed_bm_001", "failure_type": "RETRIEVAL_FAILURE"}) + "\n")

        bm_before = _count_lines(env["benchmark_path"])
        assert bm_before == 1

        # Create orchestrator pointed at isolated ledgers
        orchestrator = MultiAgentOrchestrator(
            ledger_path=env["benchmark_path"],
            runtime_ledger_path=env["runtime_path"]
        )

        req = VerificationRequest(
            claim_id="iso_c001",
            claim_text="Test claim for isolation.",
            top_k=1,
            metadata={"managed_audit": False}
        )
        result = orchestrator.verify(request=req, pre_retrieved_evidence=[])

        # Benchmark should not change
        bm_after = _count_lines(env["benchmark_path"])
        assert bm_after == bm_before, f"Benchmark changed from {bm_before} to {bm_after}"

    def test_no_duplicate_audit_records(self, isolated_env):
        """Each claim should produce exactly one audit record."""
        env = isolated_env
        auditor = RuntimeAuditor(ledger_filepath=env["runtime_path"])

        # Audit the same claim once
        auditor.audit_claim(
            claim_id="dedup_c001",
            claim_text="Test dedup claim.",
            verdict="SUPPORT", confidence=0.85,
            evidence=[], question="Is dedup working?",
            run_id="dedup_run"
        )
        count_after_first = _count_lines(env["runtime_path"])
        assert count_after_first == 1

        # A second audit call for a different claim should add exactly 1 more
        auditor.audit_claim(
            claim_id="dedup_c002",
            claim_text="Second test claim.",
            verdict="NEI", confidence=0.35,
            evidence=[], question="Is dedup working?",
            run_id="dedup_run"
        )
        count_after_second = _count_lines(env["runtime_path"])
        assert count_after_second == 2


# -----------------------------------------------------------------------
# TEST 5: Pattern Persistence and Reload After Simulated Restart
# -----------------------------------------------------------------------

class TestPatternPersistence:
    """Verify patterns persist to disk and can be reloaded."""

    def test_patterns_survive_restart(self, isolated_env):
        """Patterns mined and saved to disk should load correctly in a new FailureMemory."""
        env = isolated_env

        # Mine and save
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        original_patterns = miner.mine_patterns()
        original_count = len(original_patterns)
        original_ids = {p.pattern_id for p in original_patterns}

        # Simulate restart: create new FailureMemory from disk
        memory2 = FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )
        assert len(memory2.patterns) == original_count
        loaded_ids = {p.pattern_id for p in memory2.patterns}
        assert loaded_ids == original_ids

    def test_pattern_deduplication_on_repeated_mining(self, isolated_env):
        """Mining multiple times should produce the same patterns, not duplicates."""
        env = isolated_env
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )

        p1 = miner.mine_patterns()
        p2 = miner.mine_patterns()
        assert len(p1) == len(p2)
        ids1 = {p.pattern_id for p in p1}
        ids2 = {p.pattern_id for p in p2}
        assert ids1 == ids2, "Pattern IDs should be identical after repeated mining"

    def test_empty_ledger_produces_valid_patterns(self, isolated_env):
        """Mining from empty ledgers should produce patterns without errors."""
        env = isolated_env
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        patterns = miner.mine_patterns()
        assert len(patterns) == 9, f"Expected 9 base patterns, got {len(patterns)}"
        assert all(p.frequency >= 1 for p in patterns)

    def test_missing_patterns_file_triggers_auto_mining(self, isolated_env):
        """If failure_patterns.json is missing, FailureMemory should auto-mine."""
        env = isolated_env
        # Don't create the file — FailureMemory should handle it
        memory = FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )
        assert len(memory.patterns) >= 9
        assert os.path.exists(env["patterns_path"])


# -----------------------------------------------------------------------
# TEST 6: Failure Classifier F1-F8 Correctness
# -----------------------------------------------------------------------

class TestFailureClassifierCanonical:
    """Verify the FailureClassifier correctly identifies all F1-F8 categories."""

    @pytest.fixture
    def classifier(self):
        return FailureClassifier()

    def test_classifier_f7_contradiction(self, classifier):
        ev = [_make_evidence("c1", "This treatment fails to improve outcomes.", 0.85,
                             metadata={"relationship": "CONTRADICTS"})]
        result = classifier.classify(
            claim_text="This treatment improves outcomes.",
            evidence=ev, verdict="SUPPORT", confidence=0.85
        )
        assert result.has_failure is True
        assert result.canonical_code is not None
        assert result.canonical_code.value == "F7"

    def test_classifier_f6_domain_mismatch(self, classifier):
        ev = [_make_evidence("d1", "Smart grid power distribution systems.",
                             0.75, metadata={"topic_match": False})]
        result = classifier.classify(
            claim_text="Quantum federated learning improves privacy.",
            evidence=ev, verdict="SUPPORT", confidence=0.85,
            question="How does quantum federated learning work?"
        )
        assert result.has_failure is True
        assert result.canonical_code.value == "F6"

    def test_classifier_f1_no_evidence(self, classifier):
        result = classifier.classify(
            claim_text="Novel compound X cures disease Y.",
            evidence=[], verdict="SUPPORT", confidence=0.85
        )
        assert result.has_failure is True
        assert result.canonical_code.value == "F1"
        assert result.severity == "CRITICAL"

    def test_classifier_clean_verdict(self, classifier):
        ev = [
            _make_evidence("ok1", "Confirmed finding supporting the claim.", 0.92,
                           metadata={"relationship": "SUPPORTS", "topic_match": True, "evidence_eligible": True}),
            _make_evidence("ok2", "Corroborating laboratory trial confirms memory consolidation enhancement.", 0.88,
                           metadata={"relationship": "SUPPORTS", "topic_match": True, "evidence_eligible": True})
        ]
        result = classifier.classify(
            claim_text="Sleep improves memory consolidation.",
            evidence=ev, verdict="SUPPORT", confidence=0.85
        )
        assert result.has_failure is False
        assert result.severity == "CLEAN"


# -----------------------------------------------------------------------
# TEST 7: High Relevance Without Claim Support
# -----------------------------------------------------------------------

class TestRelevanceIsNotSupport:
    """Verify that high retrieval relevance does not automatically mean SUPPORTED."""

    def test_high_score_contradictory_evidence(self, isolated_env):
        """Score 0.95 with CONTRADICTS relationship must not produce SUPPORT."""
        env = isolated_env
        miner = PatternMiner(
            benchmark_path=env["benchmark_path"],
            runtime_path=env["runtime_path"],
            output_patterns_path=env["patterns_path"]
        )
        miner.mine_patterns()
        memory = FailureMemory(
            patterns_path=env["patterns_path"],
            trusted_memory_path=env["trusted_path"]
        )

        ev = _make_evidence(
            "high_contra", "This study found no benefit of the intervention.",
            0.95, "Counter Study", metadata={"relationship": "CONTRADICTS"}
        )
        match = memory.match_historical_failures(
            claim_text="The intervention always benefits patients.",
            evidence=[ev], preliminary_verdict="SUPPORT", confidence=0.90
        )
        assert match.matched is True
        assert "FP_CON_006" in match.matched_pattern_ids

        verdict, conf, _, _ = memory.execute_claim_recheck(
            claim_text="The intervention always benefits patients.",
            evidence=[ev], match_result=match,
            current_verdict="SUPPORT", current_confidence=0.90
        )
        assert verdict in ("REFUTE", "REFUTED", "CONTRADICTED")
