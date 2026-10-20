"""Visualization and summary data exporter utilities for benchmark evaluation."""

import os
import csv
import json
from typing import List, Dict, Any, Optional

from src.evaluation.schema import EvaluationResult


def export_baseline_comparison_csv(
    results: List[EvaluationResult],
    output_filepath: str = "results/metrics/baseline_comparison.csv"
) -> str:
    """Export comparative metrics summary across baseline systems to CSV."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    fieldnames = [
        "run_id", "system_name", "dataset", "split", "total_examples",
        "accuracy", "precision", "recall", "macro_f1", "mrr",
        "recall_at_5", "rhr_percent", "debate_activation_rate", "avg_exec_time_sec"
    ]

    rows = []
    for res in results:
        res_d = res.to_dict()
        cls_m = res_d.get("classification_metrics") or {}
        ret_m = res_d.get("retrieval_metrics") or {}
        ma_m = res_d.get("multi_agent_metrics") or {}
        rhr_m = res_d.get("rhr_metrics") or {}

        rec_5 = ret_m.get("recall_at_k", {}).get("5", 0.0)

        rows.append({
            "run_id": res.run_id,
            "system_name": res.system_name,
            "dataset": res.dataset,
            "split": res.split,
            "total_examples": res.total_examples,
            "accuracy": cls_m.get("accuracy", 0.0),
            "precision": cls_m.get("precision", 0.0),
            "recall": cls_m.get("recall", 0.0),
            "macro_f1": cls_m.get("macro_f1", 0.0),
            "mrr": ret_m.get("mrr", 0.0),
            "recall_at_5": rec_5,
            "rhr_percent": rhr_m.get("repeated_hallucination_rate", 0.0),
            "debate_activation_rate": ma_m.get("debate_activation_rate", 0.0),
            "avg_exec_time_sec": ma_m.get("average_execution_time_sec", 0.0)
        })

    with open(output_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_filepath


def export_confusion_matrix_csv(
    confusion_matrix: Dict[str, Dict[str, int]],
    output_filepath: str = "results/metrics/confusion_matrix.csv"
) -> str:
    """Export confusion matrix to CSV."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    labels = sorted(list(confusion_matrix.keys()))
    fieldnames = ["ground_truth"] + labels

    rows = []
    for gt in labels:
        row = {"ground_truth": gt}
        for pred in labels:
            row[pred] = confusion_matrix.get(gt, {}).get(pred, 0)
        rows.append(row)

    with open(output_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_filepath
