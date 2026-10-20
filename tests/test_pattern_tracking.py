"""Tests for CiteGuard Pattern Mining Delta Tracking and UI Visibility.

Verifies:
1. PatternMiner distinguishes CREATED, UPDATED, and UNCHANGED patterns.
2. Novel runtime failure types synthesize dynamic patterns (CREATED).
3. Existing patterns with new failure instances are tagged UPDATED.
4. Untouched patterns are tagged UNCHANGED.
5. FailureMemory captures and exposes the latest MiningResult.
6. API endpoint /api/ledger/patterns returns stored patterns and mining_summary.
7. API endpoint /api/verify returns pattern_mining_result.
8. Dual-ledger isolation is strictly preserved (hallucination_ledger.jsonl unchanged).
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from src.evaluation.failure_pattern import FailurePattern
from src.evaluation.pattern_miner import PatternMiner, MiningResult
from src.evaluation.failure_memory import FailureMemory
from src.ui.server import app


@pytest.fixture
def mock_tracking_env(tmp_path):
    """Fixture providing isolated file paths for pattern miner tracking."""
    runtime_path = str(tmp_path / "runtime_audit.jsonl")
    benchmark_path = str(tmp_path / "hallucination_ledger.jsonl")
    patterns_path = str(tmp_path / "failure_patterns.json")
    trusted_path = str(tmp_path / "trusted_memory.json")

    # Create empty runtime audit
    with open(runtime_path, "w", encoding="utf-8") as f:
        pass

    # Create benchmark ledger with sample record
    with open(benchmark_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "claim_id": "CLM_BENCH_001",
            "failure_type": "DOMAIN_MISMATCH",
            "verdict": "REFUTED",
            "domain": "neuroscience"
        }) + "\n")

    return {
        "runtime_path": runtime_path,
        "benchmark_path": benchmark_path,
        "patterns_path": patterns_path,
        "trusted_path": trusted_path,
        "tmp_path": tmp_path
    }


def test_pattern_miner_delta_tracking(mock_tracking_env):
    """Test that PatternMiner accurately classifies created, updated, and unchanged patterns."""
    patterns_path = mock_tracking_env["patterns_path"]
    runtime_path = mock_tracking_env["runtime_path"]
    benchmark_path = mock_tracking_env["benchmark_path"]

    miner = PatternMiner(
        ledger_path=benchmark_path,
        runtime_audit_path=runtime_path,
        output_patterns_path=patterns_path
    )

    # First run: baseline mining (all patterns are new relative to empty file)
    result1: MiningResult = miner.mine_patterns_with_tracking()
    assert len(result1.created_patterns) >= 9
    assert len(result1.updated_patterns) == 0
    assert result1.total_patterns == len(result1.patterns)

    # Immediately mine again without any new data -> all should be unchanged
    result2: MiningResult = miner.mine_patterns_with_tracking()
    assert len(result2.created_patterns) == 0
    assert len(result2.updated_patterns) == 0
    assert len(result2.unchanged_patterns) == result1.total_patterns

    # Now simulate a new runtime failure for an existing category (DOMAIN_MISMATCH -> FP_DOM_001)
    with open(runtime_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "claim_id": "CLM_RT_001",
            "claim_text": "Caffeine improves quantum teleportation.",
            "failure_type": "DOMAIN_MISMATCH",
            "domain": "physics",
            "severity": "HIGH",
            "timestamp": "2026-10-11T12:00:00Z"
        }) + "\n")

    result3: MiningResult = miner.mine_patterns_with_tracking()
    assert "FP_DOM_001" in [p["pattern_id"] for p in result3.updated_patterns]
    assert len(result3.created_patterns) == 0

    # Now simulate a novel failure type in runtime ledger not in standard templates
    with open(runtime_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "claim_id": "CLM_RT_002",
            "claim_text": "Model hallucinated author affiliation.",
            "failure_type": "AFFILIATION_FABRICATION",
            "domain": "general",
            "severity": "MEDIUM",
            "timestamp": "2026-10-11T12:05:00Z"
        }) + "\n")

    result4: MiningResult = miner.mine_patterns_with_tracking()
    created_ids = [p["pattern_id"] for p in result4.created_patterns]
    assert len(created_ids) >= 1
    assert any("AFF" in pid for pid in created_ids)


def test_failure_memory_mining_result_integration(mock_tracking_env):
    """Test that FailureMemory retains and exposes the MiningResult."""
    miner = PatternMiner(
        ledger_path=mock_tracking_env["benchmark_path"],
        runtime_audit_path=mock_tracking_env["runtime_path"],
        output_patterns_path=mock_tracking_env["patterns_path"]
    )
    miner.mine_patterns()

    memory = FailureMemory(
        patterns_path=mock_tracking_env["patterns_path"],
        trusted_memory_path=mock_tracking_env["trusted_path"],
        miner=miner
    )

    patterns_list = memory.mine_patterns_from_historical_ledger()
    assert isinstance(patterns_list, list)

    stored_res = memory.get_last_mining_result()
    assert isinstance(stored_res, MiningResult)
    assert stored_res.total_patterns == len(memory.patterns)

    as_dict = stored_res.to_dict()
    assert "created_patterns" in as_dict
    assert "updated_patterns" in as_dict
    assert "unchanged_patterns" in as_dict
    assert "total_patterns" in as_dict
    assert "timestamp" in as_dict


def test_api_ledger_patterns_endpoint():
    """Test GET /api/ledger/patterns returns complete patterns list and mining summary."""
    client = TestClient(app)
    response = client.get("/api/ledger/patterns")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "patterns" in data
    assert "total_patterns" in data
    assert data["total_patterns"] >= 9
    assert "mining_summary" in data

    summary = data["mining_summary"]
    assert "total_patterns" in summary
    assert "created_count" in summary
    assert "updated_count" in summary
    assert "unchanged_count" in summary

    # Inspect first pattern schema
    pattern = data["patterns"][0]
    assert "pattern_id" in pattern
    assert "canonical_category" in pattern
    assert "trigger_conditions" in pattern
    assert "prevention_rule" in pattern
    assert "frequency" in pattern
    assert "last_seen" in pattern
    assert "source_examples" in pattern


def test_api_verify_includes_pattern_mining_result():
    """Test POST /api/verify returns pattern_mining_result in the response."""
    client = TestClient(app)
    payload = {
        "question": "Does intermittent fasting extend human lifespan?",
        "provider": "mock",
        "top_k": 3,
        "mode": "hybrid"
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "pattern_mining_result" in data
    mining_res = data["pattern_mining_result"]
    assert "total_patterns" in mining_res
    assert "created_patterns" in mining_res
    assert "updated_patterns" in mining_res
    assert "unchanged_patterns" in mining_res


def test_dual_ledger_remains_unchanged_during_mining(mock_tracking_env):
    """Verify that mining patterns does not alter the historical benchmark ledger."""
    benchmark_path = mock_tracking_env["benchmark_path"]
    with open(benchmark_path, "r", encoding="utf-8") as f:
        initial_lines = f.readlines()

    miner = PatternMiner(
        ledger_path=benchmark_path,
        runtime_audit_path=mock_tracking_env["runtime_path"],
        output_patterns_path=mock_tracking_env["patterns_path"]
    )
    miner.mine_patterns_with_tracking()

    with open(benchmark_path, "r", encoding="utf-8") as f:
        after_lines = f.readlines()

    assert initial_lines == after_lines
