"""Dataset Statistics and Analytics module."""

import os
import json
from collections import Counter
from typing import List, Dict, Any
from src.data.schema import UnifiedClaimRecord
from src.data.loader import get_loader
from src.data.preprocessor import preprocess_records


def compute_dataset_statistics(dataset_name: str, records: List[UnifiedClaimRecord]) -> Dict[str, Any]:
    """Compute detailed dataset statistics for a given dataset."""
    total_records = len(records)
    unique_claims = len({r.claim for r in records})
    unique_evidence_ids = len({r.evidence_id for r in records if r.evidence_id})

    split_counts = Counter(r.split for r in records)
    label_counts = Counter(r.label for r in records)

    missing_evidence_count = sum(1 for r in records if not r.evidence_text)
    missing_title_count = sum(1 for r in records if not r.evidence_title)

    claim_word_lens = [len(r.claim.split()) for r in records if r.claim]
    claim_char_lens = [len(r.claim) for r in records if r.claim]

    ev_word_lens = [len(r.evidence_text.split()) for r in records if r.evidence_text]
    ev_char_lens = [len(r.evidence_text) for r in records if r.evidence_text]

    avg_claim_words = round(sum(claim_word_lens) / len(claim_word_lens), 2) if claim_word_lens else 0.0
    avg_claim_chars = round(sum(claim_char_lens) / len(claim_char_lens), 2) if claim_char_lens else 0.0

    avg_ev_words = round(sum(ev_word_lens) / len(ev_word_lens), 2) if ev_word_lens else 0.0
    avg_ev_chars = round(sum(ev_char_lens) / len(ev_char_lens), 2) if ev_char_lens else 0.0

    return {
        "dataset_name": dataset_name,
        "total_records": total_records,
        "unique_claims": unique_claims,
        "unique_evidence_documents": unique_evidence_ids,
        "split_distribution": dict(split_counts),
        "label_distribution": dict(label_counts),
        "missing_values": {
            "missing_evidence_text": missing_evidence_count,
            "missing_evidence_title": missing_title_count
        },
        "length_statistics": {
            "avg_claim_words": avg_claim_words,
            "avg_claim_chars": avg_claim_chars,
            "avg_evidence_words": avg_ev_words,
            "avg_evidence_chars": avg_ev_chars
        }
    }


def analyze_all_datasets(output_dir: str = "results/data_statistics", docs_dir: str = "docs") -> Dict[str, Any]:
    """Run full dataset statistics pipeline across SciFact, CoVERT, and SciClaimHunt."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    datasets = ["scifact", "covert", "sciclaimhunt"]
    all_stats = {}

    md_lines = [
        "# Dataset Statistics & Profiling Report",
        "",
        "This report provides machine-readable and human-readable statistics for all ingested scientific claim verification datasets.",
        ""
    ]

    for ds in datasets:
        print(f"[Analyze] Loading dataset: {ds}...")
        loader = get_loader(ds)
        raw_records = loader.load()
        processed_records, prep_stats = preprocess_records(raw_records)

        stats = compute_dataset_statistics(ds, processed_records)
        stats["preprocessing"] = prep_stats
        all_stats[ds] = stats

        md_lines.extend([
            f"## {ds.upper()} Dataset Statistics",
            f"* **Total Records**: {stats['total_records']}",
            f"* **Unique Claims**: {stats['unique_claims']}",
            f"* **Unique Evidence Documents**: {stats['unique_evidence_documents']}",
            f"* **Duplicates Removed**: {prep_stats['duplicates_removed']}",
            "",
            "### Split Distribution",
            "| Split | Count | Percentage |",
            "|:---|:---|:---|"
        ])
        for split, count in stats["split_distribution"].items():
            pct = round(count / stats['total_records'] * 100, 2) if stats['total_records'] > 0 else 0
            md_lines.append(f"| {split} | {count} | {pct}% |")

        md_lines.extend([
            "",
            "### Label Distribution",
            "| Label | Count | Percentage |",
            "|:---|:---|:---|"
        ])
        for label, count in stats["label_distribution"].items():
            pct = round(count / stats['total_records'] * 100, 2) if stats['total_records'] > 0 else 0
            md_lines.append(f"| {label} | {count} | {pct}% |")

        md_lines.extend([
            "",
            "### Text Length Statistics",
            f"* **Average Claim Length**: {stats['length_statistics']['avg_claim_words']} words ({stats['length_statistics']['avg_claim_chars']} chars)",
            f"* **Average Evidence Length**: {stats['length_statistics']['avg_evidence_words']} words ({stats['length_statistics']['avg_evidence_chars']} chars)",
            "",
            "---",
            ""
        ])

    # Save JSON statistics
    json_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[Analyze] Saved machine-readable statistics to {json_path}")

    # Save Markdown report
    md_path = os.path.join(docs_dir, "DATASET_STATISTICS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Analyze] Saved human-readable summary to {md_path}")

    return all_stats


if __name__ == "__main__":
    analyze_all_datasets()
