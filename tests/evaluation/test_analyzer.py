"""Unit tests for src/evaluation/failure_analyzer.py."""

import pytest
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_analyzer import (
    classify_retrieval_failure,
    classify_reasoning_failure,
    FailureAnalyzer
)


def test_classify_retrieval_failure():
    gt = {"doc_101", "doc_102"}
    retrieved_miss = ["doc_999", "doc_888"]
    retrieved_hit = ["doc_999", "doc_101"]

    assert classify_retrieval_failure(gt, retrieved_miss) is True
    assert classify_retrieval_failure(gt, retrieved_hit) is False


def test_classify_reasoning_failure():
    gt = {"doc_101"}
    retrieved = ["doc_101", "doc_202"]

    # Ground truth hit, but prediction is wrong
    assert classify_reasoning_failure(gt, retrieved, predicted_label="REFUTE", ground_truth_label="SUPPORT") is True

    # Prediction matches ground truth -> not a reasoning failure
    assert classify_reasoning_failure(gt, retrieved, predicted_label="SUPPORT", ground_truth_label="SUPPORT") is False


def test_failure_analyzer_correct_prediction():
    analyzer = FailureAnalyzer()
    retrieved_results = [{"doc_id": "doc_101", "rank": 1}]
    
    # Correct prediction -> returns None
    sig = analyzer.analyze_instance(
        claim_id="c1", claim_text="Claim", predicted_label="SUPPORT",
        ground_truth_label="SUPPORT", retrieved_results=retrieved_results
    )
    assert sig is None


def test_failure_analyzer_retrieval_failure():
    analyzer = FailureAnalyzer()
    retrieved_results = [{"doc_id": "doc_999", "rank": 1}]
    
    sig = analyzer.analyze_instance(
        claim_id="c2", claim_text="Claim", predicted_label="REFUTE",
        ground_truth_label="SUPPORT", retrieved_results=retrieved_results,
        ground_truth_doc_ids={"doc_101"}
    )
    assert sig is not None
    assert sig.failure_type == FailureType.RETRIEVAL_FAILURE


def test_failure_analyzer_reasoning_failure():
    analyzer = FailureAnalyzer()
    retrieved_results = [{"doc_id": "doc_101", "rank": 1}]
    
    sig = analyzer.analyze_instance(
        claim_id="c3", claim_text="Claim", predicted_label="REFUTE",
        ground_truth_label="SUPPORT", retrieved_results=retrieved_results,
        ground_truth_doc_ids={"doc_101"}
    )
    assert sig is not None
    assert sig.failure_type == FailureType.REASONING_FAILURE


def test_analyze_runtime_instance_no_evidence():
    analyzer = FailureAnalyzer()
    sig = analyzer.analyze_runtime_instance(
        claim_id="c100",
        claim_text="Test claim",
        predicted_label="NEI",
        confidence=0.9,
        evidence=[]
    )
    assert sig is not None
    assert sig.failure_type == FailureType.NO_EVIDENCE
    assert sig.signature_id.startswith("FS-")


def test_analyze_runtime_instance_ungrounded_confidence():
    analyzer = FailureAnalyzer()
    mock_ev = [{"doc_id": "d1", "score": 0.01}]
    sig = analyzer.analyze_runtime_instance(
        claim_id="c101",
        claim_text="Test claim",
        predicted_label="NEI",
        confidence=0.90,
        evidence=mock_ev
    )
    assert sig is not None
    assert sig.failure_type == FailureType.INSUFFICIENT_EVIDENCE
    assert sig.failure_subtype == "UNGROUNDED_CONFIDENCE"
    assert sig.signature_id.startswith("FS-")


def test_analyze_runtime_instance_agent_disagreement():
    analyzer = FailureAnalyzer()
    mock_ev = [{"doc_id": "d1", "score": 0.8}, {"doc_id": "d2", "score": 0.7}]
    sig = analyzer.analyze_runtime_instance(
        claim_id="c102",
        claim_text="Test claim",
        predicted_label="SUPPORTED",
        confidence=0.75,
        evidence=mock_ev,
        disagreement_status=True,
        resolution_method="debate_resolution"
    )
    assert sig is not None
    assert sig.failure_type == FailureType.REASONING_FAILURE
    assert sig.failure_subtype == "AGENT_DISAGREEMENT"

