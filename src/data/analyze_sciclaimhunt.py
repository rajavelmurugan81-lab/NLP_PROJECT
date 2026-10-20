"""Memory-Efficient Streamed Analytics script for SciClaimHunt full dataset."""

import os
import csv
import sys
import json
from typing import Dict, Any

# Increase max CSV field size limit for large scientific full-text papers
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)


def analyze_sciclaimhunt_csv(csv_path: str = "data/raw/sciclaimhunt/dataset.csv") -> Dict[str, Any]:
    """Memory-efficiently analyze full 3.18 GB dataset.csv using streaming DictReader."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SciClaimHunt CSV not found at {csv_path}")

    file_size_bytes = os.path.getsize(csv_path)
    file_size_gb = round(file_size_bytes / (1024 ** 3), 3)

    total_rows = 0
    missing_claim = 0
    missing_evidence = 0
    missing_full_paper = 0
    missing_type = 0

    seen_claims = set()
    duplicate_claims = 0

    claim_word_sum = 0
    claim_char_sum = 0
    ev_word_sum = 0
    ev_char_sum = 0
    paper_word_sum = 0
    paper_char_sum = 0

    fieldnames = []

    print(f"[SciClaimHunt Analytics] Starting streaming analysis of {csv_path} ({file_size_gb} GB)...")

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for idx, row in enumerate(reader):
            total_rows += 1

            claim = str(row.get("Claim", "")).strip()
            evidence = str(row.get("Evidence", "")).strip()
            paper = str(row.get("research_paper_full", "")).strip()
            type_val = str(row.get("Type", "")).strip()

            if not claim:
                missing_claim += 1
            if not evidence:
                missing_evidence += 1
            if not paper:
                missing_full_paper += 1
            if not type_val:
                missing_type += 1

            # Duplicate detection by claim text
            claim_lower = claim.lower()
            if claim_lower in seen_claims:
                duplicate_claims += 1
            else:
                seen_claims.add(claim_lower)

            # Length stats
            c_words = len(claim.split()) if claim else 0
            c_chars = len(claim)
            e_words = len(evidence.split()) if evidence else 0
            e_chars = len(evidence)
            p_words = len(paper.split()) if paper else 0
            p_chars = len(paper)

            claim_word_sum += c_words
            claim_char_sum += c_chars
            ev_word_sum += e_words
            ev_char_sum += e_chars
            paper_word_sum += p_words
            paper_char_sum += p_chars

            if total_rows % 25000 == 0:
                print(f"[SciClaimHunt Analytics] Processed {total_rows:,} rows...", flush=True)

    avg_claim_words = round(claim_word_sum / total_rows, 2) if total_rows > 0 else 0
    avg_claim_chars = round(claim_char_sum / total_rows, 2) if total_rows > 0 else 0

    avg_ev_words = round(ev_word_sum / total_rows, 2) if total_rows > 0 else 0
    avg_ev_chars = round(ev_char_sum / total_rows, 2) if total_rows > 0 else 0

    avg_paper_words = round(paper_word_sum / total_rows, 2) if total_rows > 0 else 0
    avg_paper_chars = round(paper_char_sum / total_rows, 2) if total_rows > 0 else 0

    results = {
        "file_path": csv_path,
        "file_size_bytes": file_size_bytes,
        "file_size_gb": file_size_gb,
        "total_records": total_rows,
        "unique_claims": len(seen_claims),
        "duplicate_claims": duplicate_claims,
        "fieldnames": fieldnames,
        "missing_values": {
            "missing_claim": missing_claim,
            "missing_evidence": missing_evidence,
            "missing_full_paper": missing_full_paper,
            "missing_type": missing_type
        },
        "length_statistics": {
            "avg_claim_words": avg_claim_words,
            "avg_claim_chars": avg_claim_chars,
            "avg_evidence_words": avg_ev_words,
            "avg_evidence_chars": avg_ev_chars,
            "avg_full_paper_words": avg_paper_words,
            "avg_full_paper_chars": avg_paper_chars
        }
    }

    return results


if __name__ == "__main__":
    res = analyze_sciclaimhunt_csv()
    print("\n=== Analysis Results ===")
    print(json.dumps(res, indent=2))
