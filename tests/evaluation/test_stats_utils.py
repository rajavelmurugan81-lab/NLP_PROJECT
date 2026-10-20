"""Unit tests for statistical significance testing and bootstrap confidence intervals."""

import pytest
from src.evaluation.schema import Prediction
from src.evaluation.stats_utils import bootstrap_confidence_interval, mcnemar_test


def test_bootstrap_confidence_interval():
    preds = [
        Prediction(example_id="1", claim_text="C1", predicted_label="SUPPORT", ground_truth_label="SUPPORT"),
        Prediction(example_id="2", claim_text="C2", predicted_label="REFUTE", ground_truth_label="REFUTE"),
        Prediction(example_id="3", claim_text="C3", predicted_label="SUPPORT", ground_truth_label="REFUTE"),
        Prediction(example_id="4", claim_text="C4", predicted_label="NEI", ground_truth_label="NEI"),
    ]

    ci = bootstrap_confidence_interval(preds, metric_name="accuracy", n_bootstraps=50, seed=42)
    assert "mean" in ci
    assert "lower" in ci
    assert "upper" in ci
    assert ci["lower"] <= ci["upper"]


def test_mcnemar_test():
    preds_a = [
        Prediction(example_id="1", claim_text="C1", predicted_label="SUPPORT", ground_truth_label="SUPPORT"),  # Correct
        Prediction(example_id="2", claim_text="C2", predicted_label="SUPPORT", ground_truth_label="REFUTE"),   # Wrong
        Prediction(example_id="3", claim_text="C3", predicted_label="REFUTE", ground_truth_label="REFUTE"),   # Correct
        Prediction(example_id="4", claim_text="C4", predicted_label="NEI", ground_truth_label="NEI"),          # Correct
    ]

    preds_b = [
        Prediction(example_id="1", claim_text="C1", predicted_label="REFUTE", ground_truth_label="SUPPORT"),   # Wrong
        Prediction(example_id="2", claim_text="C2", predicted_label="REFUTE", ground_truth_label="REFUTE"),   # Correct
        Prediction(example_id="3", claim_text="C3", predicted_label="REFUTE", ground_truth_label="REFUTE"),   # Correct
        Prediction(example_id="4", claim_text="C4", predicted_label="NEI", ground_truth_label="NEI"),          # Correct
    ]

    mcnemar_res = mcnemar_test(preds_a, preds_b)
    assert "statistic" in mcnemar_res
    assert "p_value" in mcnemar_res
    assert mcnemar_res["b10_a_right_b_wrong"] == 1
    assert mcnemar_res["b01_a_wrong_b_right"] == 1
