"""CLI runner script for Phase 7 Benchmark Evaluation & Baseline Comparison."""

import argparse
import json
import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.runner import ExperimentRunner
from src.evaluation.visualization import export_baseline_comparison_csv, export_confusion_matrix_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_evaluation")


def main():
    parser = argparse.ArgumentParser(description="CiteGuard Phase 7 Benchmark Evaluation Runner")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to config YAML")
    parser.add_argument(
        "--baseline", type=str, default="citeguard_hybrid_multiagent",
        choices=["majority_baseline", "bm25_sparse_baseline", "dense_faiss_baseline", "hybrid_rrf_baseline", "citeguard_hybrid_multiagent", "all"],
        help="System baseline to evaluate"
    )
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name ('scifact', 'covert', 'sciclaimhunt')")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('test', 'val', 'train')")
    parser.add_argument("--limit", type=int, default=10, help="Sample limit for development runs (e.g. 10 or 20)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run ID override")
    args = parser.parse_args()

    logger.info("=== Phase 7: Benchmark Evaluation Framework ===")
    logger.info(f"Dataset: {args.dataset} | Split: {args.split} | Limit: {args.limit}")

    runner = ExperimentRunner(config_filepath=args.config)

    baselines_to_run = []
    if args.baseline == "all":
        baselines_to_run = [
            "majority_baseline",
            "bm25_sparse_baseline",
            "dense_faiss_baseline",
            "hybrid_rrf_baseline",
            "citeguard_hybrid_multiagent"
        ]
    else:
        baselines_to_run = [args.baseline]

    results = []
    for sys_name in baselines_to_run:
        res = runner.run_experiment(
            system_name=sys_name,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            run_id_override=f"{args.run_id}_{sys_name}" if args.run_id else None
        )
        results.append(res)

    # Export baseline comparison table
    if len(results) > 1:
        comp_csv = export_baseline_comparison_csv(results, output_filepath="results/metrics/baseline_comparison.csv")
        logger.info(f"Exported baseline comparison CSV to: {comp_csv}")

    # Export confusion matrix for primary run
    if results and results[-1].classification_metrics:
        cm_csv = export_confusion_matrix_csv(
            results[-1].classification_metrics.confusion_matrix,
            output_filepath="results/metrics/confusion_matrix.csv"
        )
        logger.info(f"Exported confusion matrix CSV to: {cm_csv}")

    logger.info("\n=== EVALUATION RUN SUMMARY ===")
    for r in results:
        print(r.to_json())


if __name__ == "__main__":
    main()
