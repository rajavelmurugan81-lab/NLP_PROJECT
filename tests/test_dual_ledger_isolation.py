"""Comprehensive regression test suite for CiteGuard Dual-Ledger Isolation.

Validates that:
1. Live queries with detected failures log exclusively to runtime_audit.jsonl.
2. Historical benchmark ledger (hallucination_ledger.jsonl) remains strictly unchanged during live queries.
3. Clean live queries do not create spurious failure signatures.
4. Historical benchmark loader and counts remain intact and fully operational.
5. Standalone MultiAgentOrchestrator routes live runtime failures to runtime audit ledger, never benchmark ledger.
6. Benchmark evaluations (with ground_truth_label) continue to record to the benchmark ledger.
7. Zero duplicate logging occurs during end-to-end pipeline execution.
"""

import os
import json
import tempfile
import pytest

from src.verification.schema import (
    VerificationRequest,
    RetrievedEvidence,
    AgentDecision
)
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.providers.mock_provider import MockProvider
from src.research_pipeline import ResearchPipeline
from src.evaluation.hallucination_ledger import HallucinationLedger
from src.evaluation.runtime_auditor import RuntimeAuditor


def test_dual_ledger_live_failure_isolated_to_runtime_audit():
    """Live query failure (ungrounded support F1) must append to runtime_audit.jsonl
    and leave benchmark ledger (hallucination_ledger.jsonl) strictly unchanged.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        bench_file = os.path.join(tmp_dir, "hallucination_ledger.jsonl")
        runtime_file = os.path.join(tmp_dir, "runtime_audit.jsonl")

        # Pre-seed benchmark ledger with 2 reference entries
        seed_record_1 = {
            "signature_hash": "hash_bench_001",
            "claim_id": "seed_01",
            "claim_text": "Seed benchmark claim 1",
            "dataset": "scifact",
            "predicted_label": "SUPPORT",
            "ground_truth_label": "REFUTE",
            "failure_type": "CONTRADICTORY_EVIDENCE",
            "run_id": "bench_seed"
        }
        seed_record_2 = {
            "signature_hash": "hash_bench_002",
            "claim_id": "seed_02",
            "claim_text": "Seed benchmark claim 2",
            "dataset": "sciclaimhunt",
            "predicted_label": "REFUTE",
            "ground_truth_label": "SUPPORT",
            "failure_type": "REASONING_FAILURE",
            "run_id": "bench_seed"
        }
        with open(bench_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(seed_record_1) + "\n")
            f.write(json.dumps(seed_record_2) + "\n")

        # Initialize pipeline configured with isolated ledger files
        pipeline = ResearchPipeline(
            provider_type="mock",
            output_dir=os.path.join(tmp_dir, "reports"),
            runtime_audit_path=runtime_file,
            benchmark_ledger_path=bench_file
        )

        # Audit a claim with zero retrieved evidence to trigger NO_EVIDENCE (F1) failure
        req = VerificationRequest(
            claim_id="live_fail_001",
            claim_text="Unsubstantiated future premise with no empirical evidence.",
            dataset="research_question",
            metadata={"managed_audit": False}  # Standalone orchestrator audit
        )

        # Run verification via orchestrator with empty evidence
        res = pipeline.orchestrator.verify(
            request=req,
            pre_retrieved_evidence=[],
            run_id="run_live_test"
        )

        # 1. Benchmark ledger MUST NOT receive any record (still exactly 2 seeds)
        bench_ledger = HallucinationLedger(ledger_filepath=bench_file)
        assert len(bench_ledger.signatures) == 2
        with open(bench_file, "r", encoding="utf-8") as f:
            bench_lines = [l.strip() for l in f if l.strip()]
        assert len(bench_lines) == 2

        # 2. Runtime audit ledger MUST receive the failure
        assert os.path.exists(runtime_file)
        with open(runtime_file, "r", encoding="utf-8") as f:
            runtime_lines = [json.loads(l.strip()) for l in f if l.strip()]
        assert len(runtime_lines) == 1
        audit_entry = runtime_lines[0]
        assert audit_entry["claim_id"] == "live_fail_001"
        assert audit_entry["ledger_status"] == "LOGGED_TO_AUDIT"
        assert audit_entry["failure_type"] == "NO_EVIDENCE"


def test_clean_live_query_creates_no_spurious_failure_signature():
    """A clean, fully supported live query with >= 2 qualifying passages must not create
    a spurious failure signature and must not log an anomaly to either ledger.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        bench_file = os.path.join(tmp_dir, "hallucination_ledger.jsonl")
        runtime_file = os.path.join(tmp_dir, "runtime_audit.jsonl")

        orchestrator = MultiAgentOrchestrator(
            provider=MockProvider(default_decision="SUPPORT", default_confidence=0.92),
            ledger_path=bench_file,
            runtime_ledger_path=runtime_file
        )

        clean_evidence = [
            RetrievedEvidence(
                doc_id="clean_doc_01",
                text="Sample evidence passage supporting the claim directly and conclusively.",
                score=0.92,
                rank=1,
                metadata={"relationship": "SUPPORTS"}
            ),
            RetrievedEvidence(
                doc_id="clean_doc_02",
                text="Secondary clinical study validating and supporting the claim findings.",
                score=0.88,
                rank=2,
                metadata={"relationship": "SUPPORTS"}
            )
        ]

        req = VerificationRequest(
            claim_id="clean_claim_001",
            claim_text="MicroRNA-21 inhibits cell apoptosis.",
            dataset="research_question"
        )

        res = orchestrator.verify(request=req, pre_retrieved_evidence=clean_evidence, run_id="run_clean")

        # Invariant: No failure signature created
        assert res.failure_signature is None
        assert res.final_label in ("SUPPORT", "SUPPORTED")

        # Invariant: Benchmark ledger has 0 records
        if os.path.exists(bench_file):
            with open(bench_file, "r", encoding="utf-8") as f:
                assert sum(1 for l in f if l.strip()) == 0

        # Invariant: Standalone clean query with no failure does not log anomaly to runtime ledger
        if os.path.exists(runtime_file):
            with open(runtime_file, "r", encoding="utf-8") as f:
                runtime_records = [json.loads(l) for l in f if l.strip()]
                for r in runtime_records:
                    assert r.get("ledger_status") == "CLEAN" or r.get("severity") == "NONE"


def test_standalone_orchestrator_live_failure_routes_to_runtime_audit():
    """When MultiAgentOrchestrator is invoked directly (without pipeline managed_audit),
    detected live failures must route to runtime_audit.jsonl and NEVER to benchmark ledger.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        bench_file = os.path.join(tmp_dir, "benchmark_ledger.jsonl")
        runtime_file = os.path.join(tmp_dir, "runtime_audit.jsonl")

        orchestrator = MultiAgentOrchestrator(
            provider=MockProvider(default_decision="REFUTE", default_confidence=0.88),
            ledger_path=bench_file,
            runtime_ledger_path=runtime_file
        )

        # No evidence -> triggers NO_EVIDENCE runtime failure
        req = VerificationRequest(
            claim_id="standalone_fail_001",
            claim_text="Unsubstantiated assertion with zero literature backing.",
            dataset="research_question"
        )

        res = orchestrator.verify(request=req, pre_retrieved_evidence=[], run_id="run_standalone")

        # Verify failure signature was captured in result
        assert res.final_label == "NEI"
        assert res.failure_signature is not None
        assert res.failure_signature["claim_id"] == "standalone_fail_001"

        # Verify benchmark ledger was NEVER touched
        if os.path.exists(bench_file):
            with open(bench_file, "r", encoding="utf-8") as f:
                assert sum(1 for l in f if l.strip()) == 0

        # Verify runtime audit ledger received the record
        assert os.path.exists(runtime_file)
        with open(runtime_file, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) == 1
        assert records[0]["claim_id"] == "standalone_fail_001"
        assert records[0]["failure_type"] == "NO_EVIDENCE"


def test_benchmark_mode_with_ground_truth_still_records_to_benchmark_ledger():
    """When ground_truth_label is provided and verification disagrees,
    the failure MUST be recorded to the benchmark ledger (preserving Phase 5 functionality).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        bench_file = os.path.join(tmp_dir, "benchmark_ledger.jsonl")
        runtime_file = os.path.join(tmp_dir, "runtime_audit.jsonl")

        orchestrator = MultiAgentOrchestrator(
            provider=MockProvider(default_decision="REFUTE", default_confidence=0.85),
            ledger_path=bench_file,
            runtime_ledger_path=runtime_file
        )

        req = VerificationRequest(
            claim_id="benchmark_eval_001",
            claim_text="Known scientific truth.",
            dataset="scifact",
            ground_truth_label="SUPPORT",
            ground_truth_doc_ids=["doc_true"]
        )

        evidence = [
            RetrievedEvidence(doc_id="doc_true", text="Empirical support.", score=0.90, rank=1)
        ]

        res = orchestrator.verify(request=req, pre_retrieved_evidence=evidence, run_id="run_eval")

        # Predicted REFUTE vs Ground Truth SUPPORT -> Benchmark Failure Signature
        assert res.final_label == "REFUTE"
        assert res.failure_signature is not None

        # Verify benchmark ledger received the record
        assert os.path.exists(bench_file)
        with open(bench_file, "r", encoding="utf-8") as f:
            bench_records = [json.loads(l) for l in f if l.strip()]
        assert len(bench_records) == 1
        assert bench_records[0]["claim_id"] == "benchmark_eval_001"
        assert bench_records[0]["dataset"] == "scifact"

        # Runtime audit ledger was NOT written for benchmark evaluations
        if os.path.exists(runtime_file):
            with open(runtime_file, "r", encoding="utf-8") as f:
                assert sum(1 for l in f if l.strip()) == 0


def test_no_duplicate_logging_in_research_pipeline():
    """In ResearchPipeline, each verified claim must produce exactly ONE record
    in runtime_audit.jsonl, with zero duplicate logging.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        bench_file = os.path.join(tmp_dir, "benchmark_ledger.jsonl")
        runtime_file = os.path.join(tmp_dir, "runtime_audit.jsonl")

        pipeline = ResearchPipeline(
            provider_type="mock",
            output_dir=os.path.join(tmp_dir, "reports"),
            runtime_audit_path=runtime_file,
            benchmark_ledger_path=bench_file
        )
        pipeline.failure_memory = FailureMemory(
            trusted_memory_path=os.path.join(tmp_dir, "trusted_memory.json")
        )

        res = pipeline.run_pipeline(
            research_question="Does sleep deprivation impair spatial working memory in rodents?",
            top_k=3,
            mode="openalex_hybrid",
            verbose=False,
            max_claims=3
        )

        claims_count = len(res.get("grounded_claims", []))
        assert claims_count > 0

        # Benchmark ledger MUST remain empty
        if os.path.exists(bench_file):
            with open(bench_file, "r", encoding="utf-8") as f:
                assert sum(1 for l in f if l.strip()) == 0

        # Runtime audit ledger MUST have exactly claims_count records (no duplicates)
        assert os.path.exists(runtime_file)
        with open(runtime_file, "r", encoding="utf-8") as f:
            runtime_records = [json.loads(l) for l in f if l.strip()]
        assert len(runtime_records) == claims_count

        # All claim IDs must be unique in runtime audit records
        claim_ids = [r["claim_id"] for r in runtime_records]
        assert len(claim_ids) == len(set(claim_ids))


def test_historical_benchmark_loader_and_counts_intact():
    """The historical benchmark file (results/hallucination_ledger/hallucination_ledger.jsonl)
    must load cleanly without corruption and contain >= 160 historical reference records.
    """
    prod_bench_file = os.path.join("results", "hallucination_ledger", "hallucination_ledger.jsonl")
    assert os.path.exists(prod_bench_file)

    ledger = HallucinationLedger(ledger_filepath=prod_bench_file)
    count = ledger.load_ledger()

    assert count >= 160
    assert len(ledger.signatures) == count

    # Verify query capability
    scifact_sigs = ledger.query(dataset="scifact")
    assert len(scifact_sigs) > 0

    # Verify aggregation
    agg = ledger.aggregate_counts()
    assert agg["total_failures_recorded"] == count
    assert "by_failure_type" in agg
    assert "by_dataset" in agg
