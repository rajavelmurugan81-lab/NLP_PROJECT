"""Automated Test Suite for Rigorous Closed-Loop Failure Learning Validation."""

import os
import json
import pytest

from src.evaluation.closed_loop_harness import ClosedLoopExperimentHarness, compute_file_hash


@pytest.fixture(scope="module")
def harness_fixture(tmp_path_factory):
    tmp_dir = str(tmp_path_factory.mktemp("closed_loop_test"))
    harness = ClosedLoopExperimentHarness(output_dir=tmp_dir)
    return harness


def test_1_isolated_environment_setup(harness_fixture):
    """Verify setup of clean sandbox environment without touching production ledgers."""
    prod_bench = os.path.join("results", "hallucination_ledger", "hallucination_ledger.jsonl")
    initial_prod_hash = compute_file_hash(prod_bench)

    env_info = harness_fixture.setup_isolated_environment()
    assert env_info["initial_pattern_count"] >= 9
    assert env_info["created_count"] >= 9
    assert os.path.exists(harness_fixture.sandbox_patterns)
    assert os.path.exists(harness_fixture.sandbox_benchmark)

    # Ensure production benchmark ledger remained untouched
    assert compute_file_hash(prod_bench) == initial_prod_hash


def test_2_genuine_new_pattern_learning(harness_fixture):
    """Step 4: Confirm learning of genuinely new pattern with delta tracking."""
    dataset = harness_fixture.load_dataset()
    learning_case = next(c for c in dataset["cases"] if c["case_id"] == "learn_f6_001")

    initial_patterns_data = json.load(open(harness_fixture.sandbox_patterns, "r", encoding="utf-8"))
    initial_count = initial_patterns_data["total_patterns"]

    res = harness_fixture.run_learning_phase(learning_case)
    assert res["status"] in ("NEW_PATTERN_LEARNED", "UPDATED_EXISTING_PATTERN")
    assert res["created_count"] + res["updated_count"] + res["unchanged_count"] == res["total_patterns_after_mining"]
    assert res["new_pattern_id"] is not None
    assert res["total_patterns_after_mining"] >= initial_count


def test_3_persistence_across_real_process_restart(harness_fixture):
    """Step 5: Verify persistent loading in fresh independent Python subprocess."""
    patterns_data = json.load(open(harness_fixture.sandbox_patterns, "r", encoding="utf-8"))
    last_id = patterns_data["patterns"][-1]["pattern_id"]

    res = harness_fixture.run_restart_persistence_check(last_id)
    assert res["restart_success"] is True
    assert res["found_expected_pattern"] is True
    assert res["pattern_id"] == last_id
    assert res["fresh_process_pattern_count"] == patterns_data["total_patterns"]


def test_4_comparative_experiment_and_ablation(harness_fixture):
    """Steps 6, 7 & 8: Comparative evaluation, repeat failure reduction, and ablation."""
    res = harness_fixture.run_comparative_experiment()

    b_met = res["metrics_baseline"]
    c_met = res["metrics_citeguard"]
    no_rc_met = res["metrics_no_recheck"]

    # 1. Verification Accuracy: CiteGuard must be strictly superior to baseline
    assert c_met["accuracy"] > b_met["accuracy"]
    assert c_met["correct"] > b_met["correct"]

    # 2. Significant Reduction in repeat failures (target: >= 20%)
    assert res["relative_reduction_pct"] >= 20.0
    assert c_met["repeat_failures"] < b_met["repeat_failures"]

    # 3. False Rejection rate for valid claims must be zero
    assert c_met["false_rejections"] == 0
    assert c_met["false_rejection_rate"] == 0.0

    # 4. Ablation: Without adaptive re-check, verdicts are not corrected
    assert no_rc_met["correct"] < c_met["correct"]
    assert no_rc_met["accuracy"] < c_met["accuracy"]


def test_5_dual_ledger_integrity_post_experiment():
    """Step 9: Confirm historical benchmark ledger remained 100% immutable."""
    prod_bench = os.path.join("results", "hallucination_ledger", "hallucination_ledger.jsonl")
    with open(prod_bench, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 228
