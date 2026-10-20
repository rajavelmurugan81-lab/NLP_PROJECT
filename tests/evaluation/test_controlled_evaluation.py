"""Tests for Controlled F1-F8 Failure Evaluation Suite."""

import pytest
from src.evaluation.controlled_evaluation import (
    ControlledEvaluationSuite,
    ControlledFailureCase,
    ControlledEvaluationReport
)
from src.evaluation.failure_taxonomy import CanonicalFailureType
from src.verification.schema import RetrievedEvidence


def test_controlled_suite_case_generation():
    """Verify that case generation covers all canonical failure types F1 to F8."""
    suite = ControlledEvaluationSuite()
    cases = suite.generate_controlled_cases()
    assert len(cases) >= 9
    
    covered_types = {c.expected_failure_type for c in cases}
    assert CanonicalFailureType.F1 in covered_types
    assert CanonicalFailureType.F2 in covered_types
    assert CanonicalFailureType.F3 in covered_types
    assert CanonicalFailureType.F4 in covered_types
    assert CanonicalFailureType.F5 in covered_types
    assert CanonicalFailureType.F6 in covered_types
    assert CanonicalFailureType.F7 in covered_types
    assert CanonicalFailureType.F8 in covered_types


def test_controlled_evaluation_f1_detection():
    """Test controlled detection of F1 (Unsupported Claim)."""
    suite = ControlledEvaluationSuite()
    case = ControlledFailureCase(
        case_id="c_f1",
        expected_failure_type=CanonicalFailureType.F1,
        claim_text="Substance X cures aging.",
        evidence=[],
        preliminary_verdict="SUPPORTED",
        confidence=0.9,
        description="Empty evidence."
    )
    report = suite.run_evaluation([case])
    assert report.total_test_cases == 1
    assert report.detected_count == 1
    assert report.exact_match_count == 1
    assert report.detection_rate == 1.0


def test_controlled_evaluation_f3_detection():
    """Test controlled detection of F3 (Overgeneralization)."""
    suite = ControlledEvaluationSuite()
    case = ControlledFailureCase(
        case_id="c_f3",
        expected_failure_type=CanonicalFailureType.F3,
        claim_text="Sleep loss permanently damages brain cells in all human populations.",
        evidence=[
            RetrievedEvidence(doc_id="d1", text="Sleep loss reduced reaction time in 12 mice.", score=0.85, rank=1)
        ],
        preliminary_verdict="SUPPORTED",
        confidence=0.88,
        description="Universal claim from animal study."
    )
    report = suite.run_evaluation([case])
    assert report.detected_count == 1


def test_controlled_evaluation_f7_detection():
    """Test controlled detection of F7 (Contradictory Evidence)."""
    suite = ControlledEvaluationSuite()
    case = ControlledFailureCase(
        case_id="c_f7",
        expected_failure_type=CanonicalFailureType.F7,
        claim_text="Caffeine intake improves attention.",
        evidence=[
            RetrievedEvidence(doc_id="d1", text="Caffeine significantly improves sustained attention.", score=0.90, rank=1),
            RetrievedEvidence(doc_id="d2", text="Caffeine does not improve attention and causes anxiety.", score=0.88, rank=2)
        ],
        preliminary_verdict="SUPPORTED",
        confidence=0.85,
        description="Direct contradiction in evidence."
    )
    report = suite.run_evaluation([case])
    assert report.detected_count == 1


def test_controlled_evaluation_full_run():
    """Test running full controlled evaluation suite across all cases."""
    suite = ControlledEvaluationSuite()
    report = suite.run_evaluation()
    assert report.total_test_cases >= 9
    assert report.detected_count >= 8
    assert report.detection_rate >= 0.80
    assert report.exact_match_accuracy >= 0.70
    assert "F1" in report.per_category_metrics
    assert "F7" in report.per_category_metrics
