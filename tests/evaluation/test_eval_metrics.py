"""Unit tests for retrieval, classification, calibration, and RHR metric functions."""

import pytest
from src.evaluation.schema import Prediction
from src.evaluation.metrics import (
    compute_retrieval_metrics,
    compute_classification_metrics,
    compute_calibration_ece,
    compute_rhr_metrics
)
from src.evaluation.failure_signature import FailureSignature
from src.evaluation.failure_taxonomy import FailureType


def test_compute_retrieval_metrics():
    preds = [
        Prediction(example_id="1", claim_text="C1", predicted_label="SUPPORT", retrieved_doc_ids=["d1", "d2", "d3"]),
        Prediction(example_id="2", claim_text="C2", predicted_label="REFUTE", retrieved_doc_ids=["x1", "d4", "x2"]),
    ]
    gt_map = {
        "1": {"d1"},
        "2": {"d4"}
    }

    ret_m = compute_retrieval_metrics(preds, gt_map, k_list=[1, 3, 5])
    assert ret_m.mrr == (1.0 + 0.5) / 2.0  # 1st at rank 1 (1.0), 2nd at rank 2 (0.5) -> avg 0.75
    assert ret_m.hit_at_k[3] == 1.0


def test_compute_classification_metrics():
    preds = [
        Prediction(example_id="1", claim_text="C1", predicted_label="SUPPORT", ground_truth_label="SUPPORT"),
        Prediction(example_id="2", claim_text="C2", predicted_label="REFUTE", ground_truth_label="REFUTE"),
        Prediction(example_id="3", claim_text="C3", predicted_label="SUPPORT", ground_truth_label="REFUTE"),
        Prediction(example_id="4", claim_text="C4", predicted_label="NEI", ground_truth_label="NEI"),
    ]

    cls_m = compute_classification_metrics(preds, labels=["SUPPORT", "REFUTE", "NEI"])
    assert cls_m.accuracy == 0.75  # 3/4 correct
    assert cls_m.macro_f1 > 0.0
    assert "SUPPORT" in cls_m.confusion_matrix


def test_compute_calibration_ece():
    preds = [
        Prediction(example_id="1", claim_text="C1", predicted_label="SUPPORT", ground_truth_label="SUPPORT", confidence=0.9),
        Prediction(example_id="2", claim_text="C2", predicted_label="REFUTE", ground_truth_label="REFUTE", confidence=0.8),
    ]

    ece = compute_calibration_ece(preds)
    assert 0.0 <= ece <= 1.0


def test_compute_rhr_metrics():
    sig1 = FailureSignature(claim_id="c1", claim_text="T1", dataset="scifact", predicted_label="REFUTE", ground_truth_label="SUPPORT", failure_type=FailureType.REASONING_FAILURE, run_id="r1")
    sig2 = FailureSignature(claim_id="c2", claim_text="T2", dataset="scifact", predicted_label="SUPPORT", ground_truth_label="REFUTE", failure_type=FailureType.RETRIEVAL_FAILURE, run_id="r1")

    current_sigs = [sig1, sig2]
    historical_sigs = [sig1]  # sig1 repeated!

    rhr = compute_rhr_metrics(current_sigs, historical_sigs, stateless_rhr_baseline=0.50)
    assert rhr.total_failures == 2
    assert rhr.repeated_failures_count == 1
    assert rhr.repeated_hallucination_rate == 0.50
    assert rhr.rhr_reduction_percent == 0.0
