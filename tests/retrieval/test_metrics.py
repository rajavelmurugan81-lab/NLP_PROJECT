"""Unit tests for src/evaluation/retrieval_metrics.py."""

import pytest
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval_query
)


def test_recall_at_k():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    gt = {"d2", "d4"}

    assert recall_at_k(retrieved, gt, k=1) == 0.0
    assert recall_at_k(retrieved, gt, k=2) == 0.5  # d2 retrieved
    assert recall_at_k(retrieved, gt, k=5) == 1.0  # d2, d4 retrieved


def test_precision_at_k():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    gt = {"d2", "d4"}

    assert precision_at_k(retrieved, gt, k=2) == 0.5  # 1 relevant in top 2
    assert precision_at_k(retrieved, gt, k=5) == 0.4  # 2 relevant in top 5


def test_mean_reciprocal_rank():
    retrieved = ["d1", "d2", "d3"]
    gt = {"d2"}
    assert mean_reciprocal_rank(retrieved, gt) == 0.5  # d2 is at rank 2

    gt_miss = {"d99"}
    assert mean_reciprocal_rank(retrieved, gt_miss) == 0.0


def test_ndcg_at_k():
    retrieved = ["d1", "d2", "d3"]
    gt = {"d1"}
    assert ndcg_at_k(retrieved, gt, k=3) == 1.0  # Relevant item at rank 1


def test_evaluate_retrieval_query():
    retrieved_results = [
        {"doc_id": "d1"},
        {"doc_id": "d2"},
        {"doc_id": "d3"}
    ]
    gt = {"d2"}

    metrics = evaluate_retrieval_query(retrieved_results, gt, k_list=[1, 3])

    assert metrics["mrr"] == 0.5
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["precision@3"] == round(1/3, 4)
