"""Retrieval Quality Evaluation Metrics module (Recall@K, Precision@K, MRR, nDCG@K)."""

import math
from typing import List, Set, Dict, Any


def recall_at_k(retrieved_doc_ids: List[str], ground_truth_doc_ids: Set[str], k: int) -> float:
    """Calculate Recall@K: Proportion of relevant documents retrieved in top-k."""
    if not ground_truth_doc_ids:
        return 0.0

    top_k_retrieved = set(retrieved_doc_ids[:k])
    relevant_retrieved = top_k_retrieved.intersection(ground_truth_doc_ids)
    return len(relevant_retrieved) / len(ground_truth_doc_ids)


def precision_at_k(retrieved_doc_ids: List[str], ground_truth_doc_ids: Set[str], k: int) -> float:
    """Calculate Precision@K: Proportion of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0

    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_count = sum(1 for d in top_k_retrieved if d in ground_truth_doc_ids)
    return relevant_count / k


def mean_reciprocal_rank(retrieved_doc_ids: List[str], ground_truth_doc_ids: Set[str]) -> float:
    """Calculate Reciprocal Rank (RR): Reciprocal rank of the first relevant document."""
    if not ground_truth_doc_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in ground_truth_doc_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_doc_ids: List[str], ground_truth_doc_ids: Set[str], k: int) -> float:
    """Calculate nDCG@K (Normalized Discounted Cumulative Gain)."""
    if not ground_truth_doc_ids or k <= 0:
        return 0.0

    # Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], 1):
        if doc_id in ground_truth_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal Discounted Cumulative Gain (IDCG)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ground_truth_doc_ids), k) + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_query(
    retrieved_results: List[Dict[str, Any]],
    ground_truth_doc_ids: Set[str],
    k_list: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """Compute comprehensive retrieval metrics for a single query across multiple K thresholds."""
    doc_ids = [r["doc_id"] for r in retrieved_results]

    metrics = {
        "mrr": round(mean_reciprocal_rank(doc_ids, ground_truth_doc_ids), 4)
    }

    for k in k_list:
        metrics[f"recall@{k}"] = round(recall_at_k(doc_ids, ground_truth_doc_ids, k), 4)
        metrics[f"precision@{k}"] = round(precision_at_k(doc_ids, ground_truth_doc_ids, k), 4)
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(doc_ids, ground_truth_doc_ids, k), 4)

    return metrics
