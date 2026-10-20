"""Unit tests for baseline systems."""

import pytest
from src.evaluation.schema import EvaluationExample
from src.evaluation.baselines import (
    MajorityBaseline,
    SparseBM25Baseline,
    DenseFAISSBaseline,
    HybridBaseline,
    MultiAgentBaseline
)


def test_majority_baseline():
    base = MajorityBaseline(majority_label="SUPPORT")
    ex = EvaluationExample(example_id="1", claim_text="Test claim", ground_truth_label="REFUTE")
    pred = base.predict(ex)
    assert pred.predicted_label == "SUPPORT"
    assert pred.system_name == "majority_baseline"


def test_sparse_bm25_baseline_fallback():
    base = SparseBM25Baseline(bm25_retriever=None)
    ex = EvaluationExample(example_id="2", claim_text="Test claim 2")
    pred = base.predict(ex)
    assert pred.predicted_label == "NEI"
    assert pred.system_name == "bm25_sparse_baseline"


def test_multi_agent_baseline_fallback():
    base = MultiAgentBaseline(provider_type="mock")
    ex = EvaluationExample(example_id="3", claim_text="MicroRNA-21 regulates cell apoptosis.", ground_truth_label="SUPPORT")
    pred = base.predict(ex)
    assert pred.predicted_label in ("SUPPORT", "REFUTE", "NEI")
    assert pred.system_name == "citeguard_hybrid_multiagent"
