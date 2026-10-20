"""Unit tests for Phase 7 evaluation schemas."""

import pytest
from src.evaluation.schema import (
    EvaluationExample,
    Prediction,
    RetrievalMetrics,
    ClassificationMetrics,
    MultiAgentMetrics,
    RHRMetrics,
    EvaluationResult
)


def test_evaluation_example_serialization():
    ex = EvaluationExample(
        example_id="ex_001",
        claim_text="Sample scientific claim.",
        dataset="scifact",
        ground_truth_label="SUPPORT",
        ground_truth_doc_ids=["doc_1"]
    )
    d = ex.to_dict()
    assert d["example_id"] == "ex_001"
    assert d["ground_truth_label"] == "SUPPORT"


def test_prediction_serialization():
    p = Prediction(
        example_id="ex_002",
        claim_text="Sample claim 2",
        predicted_label="REFUTE",
        ground_truth_label="SUPPORT",
        confidence=0.88,
        retrieved_doc_ids=["doc_1", "doc_2"],
        retrieved_ranks=[1, 2],
        system_name="hybrid_rrf_baseline"
    )
    d = p.to_dict()
    assert d["predicted_label"] == "REFUTE"
    assert d["confidence"] == 0.88
    assert d["system_name"] == "hybrid_rrf_baseline"


def test_evaluation_result_json():
    ret_m = RetrievalMetrics(recall_at_k={5: 0.8}, mrr=0.75)
    cls_m = ClassificationMetrics(accuracy=0.85, macro_f1=0.82)
    
    res = EvaluationResult(
        run_id="eval_run_001",
        system_name="citeguard_hybrid_multiagent",
        dataset="scifact",
        split="test",
        total_examples=10,
        retrieval_metrics=ret_m,
        classification_metrics=cls_m
    )

    d = res.to_dict()
    assert d["run_id"] == "eval_run_001"
    assert d["classification_metrics"]["accuracy"] == 0.85

    json_str = res.to_json()
    assert '"eval_run_001"' in json_str
