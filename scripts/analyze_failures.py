"""Script to execute failure analysis pipeline, populate Hallucination Ledger, and export metrics."""

import os
import sys
import argparse
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid import HybridRetriever
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_analyzer import FailureAnalyzer
from src.evaluation.hallucination_ledger import HallucinationLedger
from src.evaluation.failure_statistics import generate_visualization_tables, compute_failure_statistics


def run_failure_analysis_demo():
    print("=== Phase 5: Executing Failure Analysis & Hallucination Ledger Pipeline ===")
    config = RetrievalConfig()
    retriever = HybridRetriever(config=config)

    if not retriever.load_indexes():
        print("[Error] Retrieval indexes not found. Run 'python scripts/build_indexes.py' first.")
        return

    ledger = HallucinationLedger()
    analyzer = FailureAnalyzer(default_run_id="run_phase5_demo")

    # Benchmark test evaluation queries with ground truth annotations
    test_eval_queries = [
        {
            "claim_id": "eval_001",
            "claim_text": "CRISPR-Cas9 enables targeted genome editing in eukaryotic cell lines.",
            "ground_truth_label": "SUPPORT",
            "predicted_label": "SUPPORT",
            "ground_truth_doc_ids": {"21561474"},
            "dataset": "scifact"
        },
        {
            "claim_id": "eval_002",
            "claim_text": "Vitamin D supplementation guarantees 100% immunity against viral infection.",
            "ground_truth_label": "REFUTE",
            "predicted_label": "SUPPORT",  # Simulated reasoning/contradiction failure
            "ground_truth_doc_ids": {"doc_covert_covert_001"},
            "dataset": "covert"
        },
        {
            "claim_id": "eval_003",
            "claim_text": "Quantum annealing algorithms achieve exponential speedup over classical solvers for all NP-hard problems.",
            "ground_truth_label": "REFUTE",
            "predicted_label": "NEUTRAL",  # Simulated insufficient evidence failure
            "ground_truth_doc_ids": {"doc_sch_sch_102"},
            "dataset": "sciclaimhunt"
        },
        {
            "claim_id": "eval_004",
            "claim_text": "Inhibition of nonhomologous end joining decreases genome editing efficiency.",
            "ground_truth_label": "REFUTE",
            "predicted_label": "REFUTE",
            "ground_truth_doc_ids": {"21561474"},
            "dataset": "scifact"
        },
        {
            "claim_id": "eval_005",
            "claim_text": "Novel rare genetic mutation X caused immediate therapeutic reversal.",
            "ground_truth_label": "SUPPORT",
            "predicted_label": "REFUTE",
            "ground_truth_doc_ids": {"unretrieved_doc_999"},  # Simulated retrieval failure
            "dataset": "scifact"
        }
    ]

    print(f"\nAnalyzing {len(test_eval_queries)} benchmark evaluation queries...")
    for q in test_eval_queries:
        mode = "hybrid"
        results = retriever.retrieve(q["claim_text"], mode=mode, top_k=5)
        
        sig = analyzer.analyze_instance(
            claim_id=q["claim_id"],
            claim_text=q["claim_text"],
            predicted_label=q["predicted_label"],
            ground_truth_label=q["ground_truth_label"],
            retrieved_results=results,
            ground_truth_doc_ids=q["ground_truth_doc_ids"],
            dataset=q["dataset"],
            retrieval_mode=mode,
            run_id="run_phase5_demo"
        )

        if sig:
            added = ledger.append(sig)
            print(f"  [Failure Recorded] ID: {sig.claim_id} | Type: {sig.failure_type.value} | Deduplicated: {not added}")
        else:
            print(f"  [Correct Verdict] ID: {q['claim_id']} | Label: {q['ground_truth_label']}")

    # Export CSV & Visualization Tables
    csv_file = ledger.export_csv()
    viz_files = generate_visualization_tables(ledger)

    stats = compute_failure_statistics(ledger.signatures, total_evaluated_cases=len(test_eval_queries))
    print("\n=== Phase 5 Failure Statistics Summary ===")
    print(json.dumps(stats, indent=2))
    print("\n=== Failure Analysis Pipeline Complete ===")


if __name__ == "__main__":
    run_failure_analysis_demo()
