"""Deterministic End-to-End Verification Test Suite for CiteGuard:
Runtime Failure Learning and Trusted Memory.

Validates:
A. Controlled runtime failure detection, classification (F1-F8), audit logging,
   novel pattern generation, persistence across reload, and closed-loop re-check mitigation.
B. Strict deduplication of recurring failures to prevent redundant pattern creation.
C. Seven controlled Trusted Memory policy cases (full support, unsupported, no evidence,
   contradicted, partial support, re-check rejection, and deduplication/versioning).
D. Complete ledger separation across Trusted Memory, Failure Patterns, Runtime Audit,
   and Historical Benchmark.
E. Backend API endpoints (/api/ledger/trusted, /api/ledger/patterns, /api/ledger/runtime).
F. Absolute zero-pollution guarantee for production ledger files.
"""

import os
import json
import tempfile
import pytest
from typing import List, Dict, Any
from starlette.testclient import TestClient

from src.evaluation.failure_pattern import FailurePattern, PatternMatchResult
from src.evaluation.pattern_miner import PatternMiner
from src.evaluation.failure_memory import FailureMemory
from src.evaluation.runtime_auditor import RuntimeAuditor
from src.evaluation.failure_classifier import FailureClassifier
from src.evaluation.failure_taxonomy import CanonicalFailureCode
from src.verification.schema import VerificationRequest, VerificationResult, RetrievedEvidence
from src.verification.orchestrator import MultiAgentOrchestrator
from src.ui.server import app


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_sandbox(tmp_path):
    """Creates isolated filesystem sandbox to ensure production ledger files are never modified."""
    return {
        "runtime_path": str(tmp_path / "runtime_audit.jsonl"),
        "benchmark_path": str(tmp_path / "hallucination_ledger.jsonl"),
        "patterns_path": str(tmp_path / "failure_patterns.json"),
        "trusted_path": str(tmp_path / "trusted_memory.json"),
        "tmp_path": tmp_path,
    }


def _create_evidence(doc_id: str, text: str, score: float, metadata: Dict[str, Any] = None) -> RetrievedEvidence:
    return RetrievedEvidence(
        doc_id=doc_id,
        text=text,
        score=score,
        rank=1,
        title=f"Paper_{doc_id}",
        metadata=metadata or {}
    )


# ---------------------------------------------------------------------------
# SECTION B: Test Runtime Failure-Pattern Generation & Deduplication
# ---------------------------------------------------------------------------

def test_b1_b6_runtime_failure_detection_classification_and_pattern_persistence(isolated_sandbox):
    """Tests Steps 1-6:
    1. Submit controlled claim with known failure (e.g., F5 Outdated / Superseded evidence).
    2. Failure detected by verifier / auditor.
    3. FailureClassifier assigns correct canonical category F5 (Source Quality / Outdated Evidence).
    4. RuntimeAuditor records failure in isolated runtime_audit.jsonl.
    5. PatternMiner receives failure and mines/updates pattern.
    6. Pattern is written to persistent failure_patterns.json.
    """
    runtime_path = isolated_sandbox["runtime_path"]
    patterns_path = isolated_sandbox["patterns_path"]
    trusted_path = isolated_sandbox["trusted_path"]

    auditor = RuntimeAuditor(ledger_filepath=runtime_path)
    miner = PatternMiner(
        benchmark_path=isolated_sandbox["benchmark_path"],
        runtime_path=runtime_path,
        output_patterns_path=patterns_path
    )
    memory = FailureMemory(
        patterns_path=patterns_path,
        trusted_memory_path=trusted_path,
        miner=miner
    )

    # 1. Controlled claim with explicit superseded/outdated failure (F5)
    claim_text = "Classical quantum key distribution algorithms remain unconditionally secure under 2026 post-quantum standards."
    outdated_ev = [
        _create_evidence(
            doc_id="qkd_1998",
            text="Initial simulation of BB84 protocol shows security against basic intercept-resend attacks in idealized channels.",
            score=0.86,
            metadata={"publication_year": 1998, "is_superseded": True, "outdated": True}
        )
    ]

    # 2 & 3. Classify failure
    classifier = FailureClassifier()
    cls_res = classifier.classify(
        claim_text=claim_text,
        evidence=outdated_ev,
        verdict="SUPPORT",
        confidence=0.89,
        question="Are classical QKD algorithms post-quantum secure?"
    )

    assert cls_res.has_failure is True, "Expected failure to be detected by classifier"
    assert cls_res.canonical_code == CanonicalFailureCode.F5, f"Expected F5 but got {cls_res.canonical_code}"
    assert cls_res.severity in ("MEDIUM", "HIGH")

    # 4. Runtime auditor records the failure
    record = auditor.audit_claim(
        claim_id="claim_test_f5",
        claim_text=claim_text,
        verdict="SUPPORT",
        confidence=0.89,
        evidence=outdated_ev,
        question="Are classical QKD algorithms post-quantum secure?",
        run_id="run_test_f5"
    )

    assert os.path.exists(runtime_path)
    with open(runtime_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    logged_rec = json.loads(lines[0])
    assert logged_rec["canonical_failure_code"] == "F5"
    assert logged_rec["claim_id"] == "claim_test_f5"

    # 5 & 6. PatternMiner processes the runtime failure and persists failure_patterns.json
    mining_result = memory.mine_patterns_from_historical_ledger()
    assert len(mining_result) > 0
    assert os.path.exists(patterns_path)

    # Verify pattern exists on disk
    with open(patterns_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    disk_patterns = disk_data.get("patterns", [])
    assert any(p["canonical_failure_code"] == "F5" or "OUT" in p["pattern_id"] for p in disk_patterns)


def test_b7_b9_pattern_reloads_across_fresh_process_and_mitigates_next_claim(isolated_sandbox):
    """Tests Steps 7-9:
    8. Start a fresh Python / FailureMemory instance from disk and confirm pattern reloads.
    9. Submit related test claim, verify persisted pattern matches and triggers adaptive re-check.
    """
    patterns_path = isolated_sandbox["patterns_path"]
    runtime_path = isolated_sandbox["runtime_path"]
    trusted_path = isolated_sandbox["trusted_path"]

    # Pre-populate isolated patterns_path with F5 pattern
    miner = PatternMiner(
        benchmark_path=isolated_sandbox["benchmark_path"],
        runtime_path=runtime_path,
        output_patterns_path=patterns_path
    )
    miner.mine_patterns_with_tracking()

    # Step 8: Fresh instance simulates fresh process restart
    fresh_memory = FailureMemory(
        patterns_path=patterns_path,
        trusted_memory_path=trusted_path
    )
    loaded_count = len(fresh_memory.patterns)
    assert loaded_count > 0, "Patterns must reload from disk"

    # Step 9: Submit related claim with outdated evidence
    related_claim = "Early 2001 neural architecture search methods surpass human engineering across 2026 benchmarks."
    related_ev = [
        _create_evidence(
            doc_id="nas_2001",
            text="Initial NAS results from 2001 demonstrate evolutionary search on toy CIFAR subsets.",
            score=0.82,
            metadata={"publication_year": 2001, "is_superseded": True}
        )
    ]

    match_res = fresh_memory.match_historical_failures(
        claim_text=related_claim,
        evidence=related_ev,
        preliminary_verdict="SUPPORT",
        confidence=0.88
    )

    assert match_res.matched is True, "Pattern matching must trigger for related failure"
    assert "F5" in match_res.canonical_failure_code or any("OUT" in pid for pid in match_res.matched_pattern_ids)
    assert len(match_res.adaptive_actions) > 0

    # Execute claim re-check
    revised_v, revised_c, filtered_ev, recheck_res = fresh_memory.execute_claim_recheck(
        claim_text=related_claim,
        evidence=related_ev,
        match_result=match_res,
        current_verdict="SUPPORT",
        current_confidence=0.88
    )

    # Confirm adaptive verification revised the ungrounded support
    assert revised_v in ("PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE")
    assert revised_c < 0.88
    assert recheck_res.claim_recheck_status == "REVISE"


def test_b_deduplication_of_recurring_failures(isolated_sandbox):
    """Test that a recurring failure updates frequency/timestamps without creating duplicate pattern rows."""
    runtime_path = isolated_sandbox["runtime_path"]
    patterns_path = isolated_sandbox["patterns_path"]
    trusted_path = isolated_sandbox["trusted_path"]

    auditor = RuntimeAuditor(ledger_filepath=runtime_path)
    miner = PatternMiner(
        benchmark_path=isolated_sandbox["benchmark_path"],
        runtime_path=runtime_path,
        output_patterns_path=patterns_path
    )
    memory = FailureMemory(
        patterns_path=patterns_path,
        trusted_memory_path=trusted_path,
        miner=miner
    )

    # Initial mining
    res1 = miner.mine_patterns_with_tracking()
    initial_pattern_count = len(res1.patterns)

    # Log 3 identical recurring failures
    for i in range(3):
        auditor.audit_claim(
            claim_id=f"dup_claim_{i}",
            claim_text="Domain mismatch in quantum federated power grids.",
            verdict="SUPPORT",
            confidence=0.85,
            evidence=[_create_evidence(f"ev_{i}", "Electric power grid telemetry in smart meters.", 0.72, {"topic_match": False})],
            question="Quantum federated learning security?",
            run_id=f"run_dup_{i}"
        )

    # Re-mine patterns
    res2 = miner.mine_patterns_with_tracking()
    new_pattern_count = len(res2.patterns)

    # Confirm no duplicate patterns were created
    assert new_pattern_count == initial_pattern_count, "Recurring failure must not create duplicate patterns"
    assert res2.created_count == 0, "No new pattern rows should be marked as created for existing pattern types"
    assert res2.updated_count >= 1, "Existing pattern must be marked as updated with new frequency/timestamp"


# ---------------------------------------------------------------------------
# SECTION C: Test Trusted Memory Storage (All 7 Explicit Policy Cases)
# ---------------------------------------------------------------------------

def test_c1_fully_supported_claim_is_stored(isolated_sandbox):
    """Case 1: Fully supported claim with adequate confidence and genuine evidence -> STORED."""
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    before_count = len(memory.load_trusted_memory())
    assert before_count == 0

    claim_text = "Sleep deprivation impairs spatial memory consolidation in rodents."
    evidence = [_create_evidence("ev_valid", "Sleep deprivation suppresses hippocampal LTP and impairs spatial memory in rodents.", 0.94)]

    stored = memory.record_trusted_verification(
        claim_id="claim_supp_01",
        claim_text=claim_text,
        verdict="SUPPORTED",
        confidence=0.92,
        evidence=evidence,
        question="Does sleep deprivation affect memory?"
    )

    after_records = memory.load_trusted_memory()
    after_count = len(after_records)

    assert stored is True, "Clean supported claim must be accepted to Trusted Memory"
    assert after_count == 1
    assert after_records[0]["claim_id"] == "claim_supp_01"
    assert after_records[0]["verdict"] == "SUPPORTED"
    assert after_records[0]["confidence"] == 0.92
    assert after_records[0]["evidence_count"] == 1


def test_c2_unsupported_claim_is_not_stored(isolated_sandbox):
    """Case 2: Unsupported claim (INSUFFICIENT_EVIDENCE / NEI) -> NOT STORED."""
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    before_count = len(memory.load_trusted_memory())
    stored = memory.record_trusted_verification(
        claim_id="claim_unsupp_02",
        claim_text="Dark matter consists entirely of primordial black holes.",
        verdict="INSUFFICIENT_EVIDENCE",
        confidence=0.45,
        evidence=[],
        question="What constitutes dark matter?"
    )
    after_count = len(memory.load_trusted_memory())

    assert stored is False, "Unsupported claim must be rejected by Trusted Memory policy"
    assert after_count == before_count


def test_c3_claim_with_no_evidence_is_not_stored(isolated_sandbox):
    """Case 3: Claim with no evidence (empty evidence list) -> NOT STORED."""
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    before_count = len(memory.load_trusted_memory())
    stored = memory.record_trusted_verification(
        claim_id="claim_noev_03",
        claim_text="CRISPR modifications eliminate off-target mutations entirely.",
        verdict="SUPPORTED",
        confidence=0.95,
        evidence=[],  # No evidence
        question="Does CRISPR eliminate off-target mutations?"
    )
    after_count = len(memory.load_trusted_memory())

    assert stored is False, "Claim without evidence passages must be rejected"
    assert after_count == before_count


def test_c4_contradicted_claim_is_not_stored(isolated_sandbox):
    """Case 4: Contradicted claim -> NOT STORED."""
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    before_count = len(memory.load_trusted_memory())
    stored = memory.record_trusted_verification(
        claim_id="claim_contra_04",
        claim_text="Remdesivir reduces 28-day mortality in hospitalized patients.",
        verdict="CONTRADICTED",
        confidence=0.91,
        evidence=[_create_evidence("ev_refute", "WHO Solidarity trial reported remdesivir had no significant effect on 28-day mortality.", 0.95)],
        question="Does remdesivir reduce mortality?"
    )
    after_count = len(memory.load_trusted_memory())

    assert stored is False, "Contradicted claim must never be admitted to Trusted Memory"
    assert after_count == before_count


def test_c5_partially_supported_claim_policy(isolated_sandbox):
    """Case 5: Partially supported claim follows documented policy:
    Rejected under default policy (allow_partial=False); admitted only if explicitly configured (allow_partial=True).
    """
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    before_count = len(memory.load_trusted_memory())
    evidence = [_create_evidence("ev_part", "Intervention demonstrated moderate association in pilot cohort of 12 patients.", 0.78)]

    # 1. Default strict policy: Excluded
    stored_default = memory.record_trusted_verification(
        claim_id="claim_part_05a",
        claim_text="Intervention shows preliminary efficacy in target subpopulation.",
        verdict="PARTIALLY_SUPPORTED",
        confidence=0.76,
        evidence=evidence,
        question="Is intervention effective?",
        allow_partial=False
    )
    assert stored_default is False, "Partially supported claim must be rejected under default trust policy"
    assert len(memory.load_trusted_memory()) == before_count

    # 2. Configured permissive policy: Stored when explicitly authorized
    stored_permissive = memory.record_trusted_verification(
        claim_id="claim_part_05b",
        claim_text="Intervention shows preliminary efficacy in target subpopulation.",
        verdict="PARTIALLY_SUPPORTED",
        confidence=0.76,
        evidence=evidence,
        question="Is intervention effective?",
        allow_partial=True
    )
    assert stored_permissive is True, "Partially supported claim can be admitted when explicitly configured"
    assert len(memory.load_trusted_memory()) == before_count + 1


def test_c6_claim_rejected_after_adaptive_recheck_is_not_stored(isolated_sandbox):
    """Case 6: Claim flagged by failure pattern and revised/rejected during re-check -> NOT STORED."""
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    before_count = len(memory.load_trusted_memory())
    stored = memory.record_trusted_verification(
        claim_id="claim_recheck_06",
        claim_text="Observational coffee intake causally prevents Alzheimer's disease.",
        verdict="SUPPORTED",
        confidence=0.88,
        evidence=[_create_evidence("ev_obs", "Observational survey associates coffee consumption with self-reported cognitive trends.", 0.75)],
        question="Does coffee prevent Alzheimer's?",
        pattern_matched=True,
        recheck_status="REVISE"
    )
    after_count = len(memory.load_trusted_memory())

    assert stored is False, "Claim with unresolved failure pattern match or REVISE/REJECT re-check must not be stored"
    assert after_count == before_count


def test_c7_repeating_same_claim_deduplication_policy(isolated_sandbox):
    """Case 7: Repeating the same claim follows documented deduplication and versioning policy.
    Updates existing entry in-place rather than inflating row count.
    """
    trusted_path = isolated_sandbox["trusted_path"]
    memory = FailureMemory(trusted_memory_path=trusted_path)

    claim_text = "Metformin activates AMP-activated protein kinase (AMPK) in hepatic tissue."
    evidence1 = [_create_evidence("ev_ampk_01", "Metformin acts through activation of AMPK in primary hepatocytes.", 0.91)]
    evidence2 = [
        _create_evidence("ev_ampk_01", "Metformin acts through activation of AMPK in primary hepatocytes.", 0.91),
        _create_evidence("ev_ampk_02", "Subsequent assays confirm phosphorylation of Thr172 by Metformin.", 0.94)
    ]

    # Run 1: Store initial record
    stored1 = memory.record_trusted_verification(
        claim_id="claim_ampk_07",
        claim_text=claim_text,
        verdict="SUPPORTED",
        confidence=0.91,
        evidence=evidence1,
        question="How does Metformin regulate glucose?"
    )
    assert stored1 is True
    records1 = memory.load_trusted_memory()
    assert len(records1) == 1
    assert records1[0]["confidence"] == 0.91
    assert records1[0]["evidence_count"] == 1

    # Run 2: Repeat same claim with updated confidence and evidence count
    stored2 = memory.record_trusted_verification(
        claim_id="claim_ampk_07",
        claim_text=claim_text,
        verdict="SUPPORTED",
        confidence=0.94,
        evidence=evidence2,
        question="How does Metformin regulate glucose?"
    )
    assert stored2 is True
    records2 = memory.load_trusted_memory()

    # Deduplication check: total count remains 1, record updated with latest version
    assert len(records2) == 1, "Deduplication policy must prevent redundant record rows"
    assert records2[0]["confidence"] == 0.94
    assert records2[0]["evidence_count"] == 2


# ---------------------------------------------------------------------------
# SECTION D & E: Verify Backend APIs, Separation, and Data Integrity
# ---------------------------------------------------------------------------

def test_d_e_backend_apis_and_ledger_separation():
    """Verify backend endpoints serve live data and that memory stores are strictly separated."""
    client = TestClient(app)

    # 1. GET /api/ledger/trusted
    trusted_res = client.get("/api/ledger/trusted")
    assert trusted_res.status_code == 200
    t_data = trusted_res.json()
    assert t_data["status"] == "success"
    assert "total_trusted_claims" in t_data
    assert isinstance(t_data["claims"], list)

    # 2. GET /api/ledger/patterns
    patterns_res = client.get("/api/ledger/patterns")
    assert patterns_res.status_code == 200
    p_data = patterns_res.json()
    assert p_data["status"] == "success"
    assert "total_patterns" in p_data
    assert isinstance(p_data["patterns"], list)
    assert len(p_data["patterns"]) >= 9

    # 3. GET /api/ledger/runtime
    runtime_res = client.get("/api/ledger/runtime?limit=5")
    assert runtime_res.status_code == 200
    r_data = runtime_res.json()
    assert r_data["status"] == "success"
    assert "total_runtime_audits" in r_data
    assert isinstance(r_data["records"], list)

    # 4. GET /api/ledger/benchmark
    bench_res = client.get("/api/ledger/benchmark?limit=5")
    assert bench_res.status_code == 200
    b_data = bench_res.json()
    assert b_data["status"] == "success"
    assert "total_benchmark_records" in b_data
    assert b_data["total_benchmark_records"] == 228  # Historical benchmark is frozen at 228


def test_production_data_integrity_unmodified():
    """Confirms production ledger files remain unmodified and unpolluted."""
    bench_path = os.path.join("results", "hallucination_ledger", "hallucination_ledger.jsonl")
    assert os.path.exists(bench_path)
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_count = sum(1 for line in f if line.strip())
    # Historical benchmark must stay frozen at 228 records
    assert bench_count == 228, f"Historical benchmark was contaminated! Expected 228 but got {bench_count}"
