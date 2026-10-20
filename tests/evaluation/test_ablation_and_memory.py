"""Tests for Ablation Study and Failure-Pattern Memory Evaluation Suites."""

import pytest
from src.evaluation.ablation_study import AblationStudyRunner, AblationConfig, AblationMetrics
from src.evaluation.memory_evaluation import MemoryEvaluationRunner, MemoryEvaluationReport


def test_ablation_configurations_definition():
    """Verify 5 standard ablation configurations are defined."""
    runner = AblationStudyRunner()
    configs = runner.get_standard_configurations()
    assert len(configs) == 5
    cfg_names = [c.name for c in configs]
    assert "1_baseline_rag" in cfg_names
    assert "2_rag_multiagent" in cfg_names
    assert "3_rag_multiagent_skeptic" in cfg_names
    assert "4_rag_multiagent_memory" in cfg_names
    assert "5_citeguard_full" in cfg_names


def test_ablation_evaluation_execution():
    """Test evaluating a single ablation configuration against controlled cases."""
    runner = AblationStudyRunner()
    cases = runner.controlled_suite.generate_controlled_cases()[:3]
    cfg = runner.get_standard_configurations()[0]  # Baseline RAG
    metrics = runner.evaluate_configuration(cfg, cases)
    
    assert isinstance(metrics, AblationMetrics)
    assert metrics.config_name == "1_baseline_rag"
    assert metrics.total_evaluated == 3
    assert 0.0 <= metrics.accuracy <= 1.0


def test_ablation_full_study_output():
    """Test full ablation study output dictionary structure."""
    runner = AblationStudyRunner()
    # Mock / slice cases for fast unit test
    cases = runner.controlled_suite.generate_controlled_cases()[:2]
    configs = runner.get_standard_configurations()[:2]
    
    results = {}
    for cfg in configs:
        m = runner.evaluate_configuration(cfg, cases)
        results[cfg.name] = m.to_dict()

    assert len(results) == 2
    for name, res in results.items():
        assert "accuracy" in res
        assert "hallucination_detection_rate" in res
        assert "false_acceptance_rate" in res


def test_memory_evaluation_runner():
    """Test MemoryEvaluationRunner measuring failure reduction via pattern memory."""
    runner = MemoryEvaluationRunner()
    report = runner.run_memory_evaluation(repeat_rounds=2)
    
    assert isinstance(report, MemoryEvaluationReport)
    assert report.repeated_trials_count == 2
    assert report.total_evaluations > 0
    assert report.prevention_triggers_count >= 0
    assert report.recurring_failure_rate_without_memory == 1.0
    assert report.recurring_failure_rate_with_memory <= report.recurring_failure_rate_without_memory
    assert report.relative_failure_reduction_percent >= 0.0
