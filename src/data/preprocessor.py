"""Data Preprocessor for Scientific Claim Verification datasets."""

import re
import html
import unicodedata
from typing import List, Tuple, Dict, Any, Set
from src.data.schema import UnifiedClaimRecord


def normalize_text(text: str) -> str:
    """Safely normalize text by cleaning formatting artifacts while preserving scientific terms.

    - Decodes HTML entities (&amp; -> &, &lt; -> <)
    - Removes zero-width and invisible control characters
    - Normalizes unicode strings (NFC)
    - Normalizes whitespace (multiple spaces/newlines into single space)
    - Preserves case, chemical formulas, p-values, math expressions, and scientific punctuation
    """
    if not text:
        return ""

    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Normalize Unicode (NFC representation)
    text = unicodedata.normalize("NFC", text)

    # 3. Remove zero-width & non-printable control characters (except standard whitespace)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200d\ufeff]", "", text)

    # 4. Normalize spaces and linebreaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_formatting_artifacts(text: str) -> str:
    """Remove web/journal formatting noise like inline HTML tags or line-wrap hyphens."""
    if not text:
        return ""

    # Strip HTML tags if any present
    text = re.sub(r"<[^>]+>", "", text)

    # Remove repeated citation brackets like [1, 2, 3] or [1]-[4] if isolated at end of sentence
    text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)

    return normalize_text(text)


def detect_duplicates(records: List[UnifiedClaimRecord]) -> Tuple[List[UnifiedClaimRecord], int]:
    """Detect and remove duplicate claim-evidence records while preserving unique metadata.

    Returns deduplicated records and duplicate count.
    """
    seen_keys: Set[Tuple[str, str, str]] = set()
    deduped_records: List[UnifiedClaimRecord] = []
    duplicate_count = 0

    for rec in records:
        # Key based on normalized claim text + evidence text + source dataset
        norm_claim = rec.claim.lower().strip()
        norm_ev = (rec.evidence_text or "").lower().strip()
        key = (norm_claim, norm_ev, rec.source_dataset)

        if key in seen_keys:
            duplicate_count += 1
        else:
            seen_keys.add(key)
            deduped_records.append(rec)

    return deduped_records, duplicate_count


def validate_missing_values(records: List[UnifiedClaimRecord]) -> List[UnifiedClaimRecord]:
    """Validate and sanitize missing optional fields safely."""
    sanitized: List[UnifiedClaimRecord] = []
    for rec in records:
        rec.claim = normalize_text(rec.claim)
        if rec.evidence_text:
            rec.evidence_text = normalize_text(rec.evidence_text)
        if rec.evidence_title:
            rec.evidence_title = normalize_text(rec.evidence_title)
        sanitized.append(rec)
    return sanitized


def preprocess_records(records: List[UnifiedClaimRecord]) -> Tuple[List[UnifiedClaimRecord], Dict[str, Any]]:
    """Execute complete conservative preprocessing pipeline on unified dataset records.

    Returns preprocessed records and pipeline statistics.
    """
    initial_count = len(records)

    # 1. Normalize text and remove formatting artifacts
    cleaned_records: List[UnifiedClaimRecord] = []
    for rec in records:
        rec.claim = remove_formatting_artifacts(rec.claim)
        if rec.evidence_text:
            rec.evidence_text = remove_formatting_artifacts(rec.evidence_text)
        if rec.evidence_title:
            rec.evidence_title = remove_formatting_artifacts(rec.evidence_title)
        cleaned_records.append(rec)

    # 2. Sanitize missing fields
    sanitized_records = validate_missing_values(cleaned_records)

    # 3. Detect and remove exact duplicates
    final_records, duplicate_count = detect_duplicates(sanitized_records)

    stats = {
        "initial_count": initial_count,
        "processed_count": len(final_records),
        "duplicates_removed": duplicate_count,
        "missing_evidence_count": sum(1 for r in final_records if not r.evidence_text)
    }

    return final_records, stats
