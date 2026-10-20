"""Persistent Hallucination Ledger module for storing, querying, and exporting failure signatures."""

import os
import csv
import json
from typing import List, Dict, Any, Optional, Set
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_signature import FailureSignature

DEFAULT_LEDGER_DIR = "results/hallucination_ledger"
DEFAULT_LEDGER_FILE = os.path.join(DEFAULT_LEDGER_DIR, "hallucination_ledger.jsonl")


class HallucinationLedger:
    """Persistent storage engine for FailureSignatures with deduplication, querying, and export."""

    def __init__(self, ledger_filepath: str = DEFAULT_LEDGER_FILE):
        self.ledger_filepath = ledger_filepath
        self.signatures: List[FailureSignature] = []
        self.seen_hashes: Set[str] = set()
        os.makedirs(os.path.dirname(self.ledger_filepath), exist_ok=True)
        self.load_ledger()

    def load_ledger(self) -> int:
        """Load existing failure signatures from JSONL disk storage."""
        self.signatures = []
        self.seen_hashes = set()

        if not os.path.exists(self.ledger_filepath):
            return 0

        with open(self.ledger_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sig = FailureSignature.from_dict(data)
                    sig_hash = sig.compute_signature_hash()
                    if sig_hash not in self.seen_hashes:
                        self.seen_hashes.add(sig_hash)
                        self.signatures.append(sig)
                except Exception as e:
                    print(f"[HallucinationLedger] Warning skipping malformed line: {e}")

        return len(self.signatures)

    def append(self, signature: FailureSignature) -> bool:
        """Append a new failure signature to ledger avoiding duplicates."""
        sig_hash = signature.compute_signature_hash()
        if sig_hash in self.seen_hashes:
            return False  # Duplicate entry prevented

        self.seen_hashes.add(sig_hash)
        self.signatures.append(signature)

        # Append to JSONL file on disk
        with open(self.ledger_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(signature.to_dict()) + "\n")

        return True

    def add_signature(self, signature: FailureSignature) -> bool:
        """Alias for append method to maintain API consistency."""
        return self.append(signature)

    def append_batch(self, signatures: List[FailureSignature]) -> int:
        """Append multiple failure signatures to ledger avoiding duplicates."""
        added_count = 0
        for sig in signatures:
            if self.append(sig):
                added_count += 1
        return added_count

    def query(
        self,
        dataset: Optional[str] = None,
        failure_type: Optional[FailureType] = None,
        retrieval_mode: Optional[str] = None,
        predicted_label: Optional[str] = None,
        ground_truth_label: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[FailureSignature]:
        """Query and filter ledger signatures by attributes."""
        results = []
        for sig in self.signatures:
            if dataset and sig.dataset.lower() != dataset.lower():
                continue
            if failure_type and sig.failure_type != failure_type:
                continue
            if retrieval_mode and sig.retrieval_mode.lower() != retrieval_mode.lower():
                continue
            if predicted_label and sig.predicted_label.upper() != predicted_label.upper():
                continue
            if ground_truth_label and sig.ground_truth_label.upper() != ground_truth_label.upper():
                continue
            if run_id and sig.run_id != run_id:
                continue
            results.append(sig)
        return results

    def aggregate_counts(self) -> Dict[str, Any]:
        """Aggregate failure signature counts by dataset, type, and retrieval mode."""
        type_counts: Dict[str, int] = {}
        dataset_counts: Dict[str, int] = {}
        mode_counts: Dict[str, int] = {}

        for sig in self.signatures:
            ft_name = sig.failure_type.value
            type_counts[ft_name] = type_counts.get(ft_name, 0) + 1
            dataset_counts[sig.dataset] = dataset_counts.get(sig.dataset, 0) + 1
            mode_counts[sig.retrieval_mode] = mode_counts.get(sig.retrieval_mode, 0) + 1

        return {
            "total_failures_recorded": len(self.signatures),
            "by_failure_type": type_counts,
            "by_dataset": dataset_counts,
            "by_retrieval_mode": mode_counts
        }

    def export_csv(self, csv_filepath: Optional[str] = None) -> str:
        """Export ledger contents to CSV file."""
        target_csv = csv_filepath or os.path.join(os.path.dirname(self.ledger_filepath), "hallucination_ledger.csv")
        os.makedirs(os.path.dirname(target_csv), exist_ok=True)

        fieldnames = [
            "signature_hash", "run_id", "timestamp", "dataset", "split", "claim_id",
            "claim_text", "predicted_label", "ground_truth_label", "failure_type",
            "failure_subtype", "retrieval_mode", "confidence", "explanation"
        ]

        with open(target_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sig in self.signatures:
                d = sig.to_dict()
                row = {k: d.get(k, "") for k in fieldnames}
                writer.writerow(row)

        print(f"[HallucinationLedger] Exported {len(self.signatures):,} signatures to CSV at {target_csv}")
        return target_csv
