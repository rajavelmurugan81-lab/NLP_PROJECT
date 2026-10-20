"""Evaluation metrics calculation module for retrieval, classification, calibration, and RHR."""

import math
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

from src.evaluation.schema import (
    Prediction,
    RetrievalMetrics,
    ClassificationMetrics,
    MultiAgentMetrics,
    RHRMetrics
)
from src.evaluation.failure_signature import FailureSignature


def compute_retrieval_metrics(
    predictions: List[Prediction],
    ground_truth_map: Dict[str, Set[str]],
    k_list: List[int] = [1, 3, 5, 10]
) -> RetrievalMetrics:
    """Compute Recall@K, Precision@K, Hit@K, MRR, and nDCG@K metrics."""
    if not predictions or not ground_truth_map:
        return RetrievalMetrics()

    valid_count = 0
    recalls = defaultdict(float)
    precisions = defaultdict(float)
    hits = defaultdict(float)
    ndcgs = defaultdict(float)
    mrr_sum = 0.0

    for pred in predictions:
        gt_set = ground_truth_map.get(pred.example_id, set())
        if not gt_set:
            continue
        valid_count += 1

        retrieved = pred.retrieved_doc_ids

        # MRR calculation
        first_hit_rank = 0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in gt_set:
                first_hit_rank = rank
                break
        if first_hit_rank > 0:
            mrr_sum += 1.0 / first_hit_rank

        # Metrics @ K
        for k in k_list:
            top_k_docs = set(retrieved[:k])
            overlap = top_k_docs.intersection(gt_set)
            
            # Recall@K
            recalls[k] += len(overlap) / float(len(gt_set)) if gt_set else 0.0
            
            # Precision@K
            precisions[k] += len(overlap) / float(k)
            
            # Hit@K
            hits[k] += 1.0 if len(overlap) > 0 else 0.0

            # nDCG@K
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved[:k], 1):
                if doc_id in gt_set:
                    dcg += 1.0 / math.log2(rank + 1)
            
            ideal_hits = min(len(gt_set), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
            ndcgs[k] += (dcg / idcg) if idcg > 0 else 0.0

    if valid_count == 0:
        return RetrievalMetrics()

    return RetrievalMetrics(
        recall_at_k={k: recalls[k] / valid_count for k in k_list},
        precision_at_k={k: precisions[k] / valid_count for k in k_list},
        mrr=mrr_sum / valid_count,
        hit_at_k={k: hits[k] / valid_count for k in k_list},
        ndcg_at_k={k: ndcgs[k] / valid_count for k in k_list}
    )


def compute_classification_metrics(
    predictions: List[Prediction],
    labels: Optional[List[str]] = None
) -> ClassificationMetrics:
    """Compute Accuracy, Precision, Recall, Macro-F1, Micro-F1, and Confusion Matrix."""
    if not predictions:
        return ClassificationMetrics()

    labels = labels or ["SUPPORT", "REFUTE", "NEI"]
    labels_norm = [l.upper().strip() for l in labels]

    # Map label synonyms
    def normalize_label(lbl: str) -> str:
        u = str(lbl).strip().upper()
        if u in ("SUPPORTED", "SUPPORTS"):
            return "SUPPORT"
        elif u in ("REFUTED", "REFUTES", "CONTRADICT"):
            return "REFUTE"
        elif u in ("NEUTRAL", "INSUFFICIENT"):
            return "NEI"
        return u

    correct_count = 0
    total_count = len(predictions)

    confusion: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in labels_norm} for gt in labels_norm
    }

    for p in predictions:
        gt = normalize_label(p.ground_truth_label)
        pred = normalize_label(p.predicted_label)

        if gt not in confusion:
            confusion[gt] = {lbl: 0 for lbl in labels_norm}
        if pred not in confusion[gt]:
            confusion[gt][pred] = 0

        confusion[gt][pred] += 1

        if gt == pred:
            correct_count += 1

    accuracy = correct_count / float(total_count) if total_count > 0 else 0.0

    # Per-class Metrics
    class_report = {}
    f1_list = []
    support_list = []
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    for lbl in labels_norm:
        tp = confusion.get(lbl, {}).get(lbl, 0)
        fp = sum(confusion.get(other, {}).get(lbl, 0) for other in confusion if other != lbl)
        fn = sum(confusion.get(lbl, {}).get(other, 0) for other in labels_norm if other != lbl)
        support = sum(confusion.get(lbl, {}).values())

        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_report[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        f1_list.append(f1)
        support_list.append(support)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    macro_f1 = sum(f1_list) / float(len(f1_list)) if f1_list else 0.0
    
    micro_p = tp_sum / float(tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / float(tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    total_supp = sum(support_list)
    weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / float(total_supp) if total_supp > 0 else 0.0

    prec_avg = sum(r["precision"] for r in class_report.values()) / float(len(class_report))
    rec_avg = sum(r["recall"] for r in class_report.values()) / float(len(class_report))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=prec_avg,
        recall=rec_avg,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=confusion,
        class_report=class_report
    )


def compute_calibration_ece(predictions: List[Prediction], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    if not predictions:
        return 0.0

    bin_boundaries = [i / float(n_bins) for i in range(n_bins + 1)]
    ece = 0.0
    total_samples = len(predictions)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Extract predictions falling into [bin_lower, bin_upper)
        bin_preds = [
            p for p in predictions
            if (bin_lower <= p.confidence < bin_upper) or (i == n_bins - 1 and p.confidence == 1.0)
        ]

        if not bin_preds:
            continue

        bin_size = len(bin_preds)
        bin_acc = sum(1.0 for p in bin_preds if p.predicted_label.upper() == p.ground_truth_label.upper()) / float(bin_size)
        bin_conf = sum(p.confidence for p in bin_preds) / float(bin_size)

        ece += (bin_size / float(total_samples)) * abs(bin_acc - bin_conf)

    return round(ece, 4)


def compute_rhr_metrics(
    current_signatures: List[FailureSignature],
    historical_signatures: List[FailureSignature],
    stateless_rhr_baseline: float = 0.50
) -> RHRMetrics:
    """Calculate Repeated Hallucination Rate (RHR) and RHR Reduction (%)."""
    if not current_signatures:
        return RHRMetrics(
            repeated_hallucination_rate=0.0,
            stateless_rhr=stateless_rhr_baseline,
            rhr_reduction_percent=100.0,
            total_failures=0,
            repeated_failures_count=0
        )

    historical_hashes = {sig.compute_signature_hash() for sig in historical_signatures}

    repeated_count = 0
    for sig in current_signatures:
        if sig.compute_signature_hash() in historical_hashes:
            repeated_count += 1

    total_failures = len(current_signatures)
    rhr = repeated_count / float(total_failures) if total_failures > 0 else 0.0

    # RHR Reduction = (Stateless_RHR - CiteGuard_RHR) / Stateless_RHR * 100%
    if stateless_rhr_baseline > 0:
        reduction = max(0.0, ((stateless_rhr_baseline - rhr) / stateless_rhr_baseline) * 100.0)
    else:
        reduction = 0.0

    return RHRMetrics(
        repeated_hallucination_rate=rhr,
        stateless_rhr=stateless_rhr_baseline,
        rhr_reduction_percent=reduction,
        total_failures=total_failures,
        repeated_failures_count=repeated_count
    )
