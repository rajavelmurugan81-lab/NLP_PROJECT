"""Data Validator for Unified Claim Verification datasets."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from src.data.schema import UnifiedClaimRecord

VALID_LABELS = {
    "SUPPORT",
    "REFUTE",
    "NEUTRAL",
    "PARTIALLY_SUPPORTED",
    "INSUFFICIENT_EVIDENCE"
}

VALID_SPLITS = {"train", "dev", "test", "val"}


@dataclass
class ValidationReport:
    """Structure holding validation results and error logs."""
    is_valid: bool
    total_records: int
    valid_records_count: int
    empty_claims_count: int
    empty_evidence_count: int
    duplicate_id_count: int
    invalid_label_count: int
    invalid_split_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "valid_records_count": self.valid_records_count,
            "empty_claims_count": self.empty_claims_count,
            "empty_evidence_count": self.empty_evidence_count,
            "duplicate_id_count": self.duplicate_id_count,
            "invalid_label_count": self.invalid_label_count,
            "invalid_split_count": self.invalid_split_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors[:20],  # Truncate to top 20
            "warnings": self.warnings[:20]
        }


class DatasetValidator:
    """Validator class for verifying unified claim datasets."""

    def __init__(self, allowed_labels: Set[str] = VALID_LABELS, allowed_splits: Set[str] = VALID_SPLITS):
        self.allowed_labels = allowed_labels
        self.allowed_splits = allowed_splits

    def validate(self, records: List[UnifiedClaimRecord]) -> ValidationReport:
        """Perform comprehensive validation checks on a list of UnifiedClaimRecords."""
        total = len(records)
        errors: List[str] = []
        warnings: List[str] = []

        seen_ids: Set[str] = set()
        duplicate_id_count = 0
        empty_claims_count = 0
        empty_evidence_count = 0
        invalid_label_count = 0
        invalid_split_count = 0

        valid_count = 0

        for idx, rec in enumerate(records):
            has_error = False

            # 1. Check claim_id uniqueness and non-emptiness
            if not rec.claim_id or not rec.claim_id.strip():
                errors.append(f"Record index {idx}: Empty claim_id")
                has_error = True
            elif rec.claim_id in seen_ids:
                duplicate_id_count += 1
                warnings.append(f"Record index {idx}: Duplicate claim_id '{rec.claim_id}'")
            else:
                seen_ids.add(rec.claim_id)

            # 2. Check claim non-emptiness
            if not rec.claim or not rec.claim.strip():
                empty_claims_count += 1
                errors.append(f"Record '{rec.claim_id}': Claim text is empty")
                has_error = True

            # 3. Check evidence text (Warning if missing)
            if not rec.evidence_text or not rec.evidence_text.strip():
                empty_evidence_count += 1
                warnings.append(f"Record '{rec.claim_id}': Evidence text is empty or missing")

            # 4. Check label validity
            label_upper = rec.label.upper() if rec.label else ""
            if label_upper not in self.allowed_labels:
                invalid_label_count += 1
                errors.append(f"Record '{rec.claim_id}': Invalid label '{rec.label}'")
                has_error = True

            # 5. Check split validity
            split_lower = rec.split.lower() if rec.split else ""
            if split_lower not in self.allowed_splits:
                invalid_split_count += 1
                errors.append(f"Record '{rec.claim_id}': Invalid split name '{rec.split}'")
                has_error = True

            if not has_error:
                valid_count += 1

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            total_records=total,
            valid_records_count=valid_count,
            empty_claims_count=empty_claims_count,
            empty_evidence_count=empty_evidence_count,
            duplicate_id_count=duplicate_id_count,
            invalid_label_count=invalid_label_count,
            invalid_split_count=invalid_split_count,
            errors=errors,
            warnings=warnings
        )
