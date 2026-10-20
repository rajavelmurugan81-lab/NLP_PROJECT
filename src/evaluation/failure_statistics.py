"""Failure Statistics and Visualization Data Exporter module."""

import os
import json
from collections import Counter
from typing import List, Dict, Any, Optional
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_signature import FailureSignature
from src.evaluation.hallucination_ledger import HallucinationLedger


def compute_failure_statistics(
    signatures: List[FailureSignature],
    total_evaluated_cases: Optional[int] = None
) -> Dict[str, Any]:
    """Compute detailed failure rate statistics and taxonomy breakdowns."""
    failure_count = len(signatures)
    total_cases = total_evaluated_cases if (total_evaluated_cases and total_evaluated_cases >= failure_count) else failure_count
    correct_cases = total_cases - failure_count
    failure_pct = round((failure_count / total_cases * 100), 2) if total_cases > 0 else 0.0

    type_counts = Counter(sig.failure_type.value for sig in signatures)
    dataset_counts = Counter(sig.dataset for sig in signatures)
    label_counts = Counter(sig.ground_truth_label for sig in signatures)
    mode_counts = Counter(sig.retrieval_mode for sig in signatures)

    retrieval_failures = type_counts.get(FailureType.RETRIEVAL_FAILURE.value, 0)
    reasoning_failures = type_counts.get(FailureType.REASONING_FAILURE.value, 0)
    unknown_failures = type_counts.get(FailureType.UNKNOWN_FAILURE.value, 0)

    return {
        "total_evaluated_cases": total_cases,
        "correct_cases": correct_cases,
        "incorrect_cases": failure_count,
        "failure_count": failure_count,
        "failure_percentage": failure_pct,
        "specific_counts": {
            "retrieval_failures": retrieval_failures,
            "reasoning_failures": reasoning_failures,
            "unknown_failures": unknown_failures
        },
        "breakdown_by_failure_type": dict(type_counts),
        "breakdown_by_dataset": dict(dataset_counts),
        "breakdown_by_ground_truth_label": dict(label_counts),
        "breakdown_by_retrieval_mode": dict(mode_counts)
    }


def generate_visualization_tables(
    ledger: HallucinationLedger,
    output_dir: str = "results/hallucination_ledger"
) -> Dict[str, str]:
    """Generate machine-readable JSON tables for future web UI visualization (Task 7)."""
    os.makedirs(output_dir, exist_ok=True)
    stats = compute_failure_statistics(ledger.signatures)

    # 1. Failure type distribution table
    ft_file = os.path.join(output_dir, "failure_type_distribution.json")
    with open(ft_file, "w", encoding="utf-8") as f:
        json.dump(stats["breakdown_by_failure_type"], f, indent=2)

    # 2. Dataset-wise distribution table
    ds_file = os.path.join(output_dir, "dataset_failure_distribution.json")
    with open(ds_file, "w", encoding="utf-8") as f:
        json.dump(stats["breakdown_by_dataset"], f, indent=2)

    # 3. Retrieval-mode comparison table
    rm_file = os.path.join(output_dir, "retrieval_mode_comparison.json")
    with open(rm_file, "w", encoding="utf-8") as f:
        json.dump(stats["breakdown_by_retrieval_mode"], f, indent=2)

    # 4. Overall statistics summary
    stats_file = os.path.join(output_dir, "overall_failure_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[FailureStatistics] Exported visualization tables to {output_dir}")
    return {
        "failure_type_distribution": ft_file,
        "dataset_failure_distribution": ds_file,
        "retrieval_mode_comparison": rm_file,
        "overall_failure_statistics": stats_file
    }
