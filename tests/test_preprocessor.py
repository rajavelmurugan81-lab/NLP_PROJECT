"""Unit tests for src/data/preprocessor.py."""

import pytest
from src.data.schema import UnifiedClaimRecord
from src.data.preprocessor import (
    normalize_text,
    remove_formatting_artifacts,
    detect_duplicates,
    preprocess_records
)


def test_normalize_text_basic():
    raw = "  This is   a scientific claim  with   spaces.  "
    expected = "This is a scientific claim with spaces."
    assert normalize_text(raw) == expected


def test_normalize_text_html_and_control_chars():
    raw = "Claim with &amp; HTML entities &lt;p&gt; and \u200b zero width spaces."
    expected = "Claim with & HTML entities <p> and zero width spaces."
    assert normalize_text(raw) == expected


def test_remove_formatting_artifacts():
    raw = "<p>mRNA vaccines are effective [1, 2, 3].</p>"
    cleaned = remove_formatting_artifacts(raw)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "[1, 2, 3]" not in cleaned
    assert "mRNA vaccines are effective." in cleaned


def test_scientific_terms_preservation():
    raw = "The p-value was p < 0.001 for H2SO4 acid concentration range 10-20%."
    cleaned = normalize_text(raw)
    assert "p < 0.001" in cleaned
    assert "H2SO4" in cleaned


def test_detect_duplicates():
    r1 = UnifiedClaimRecord(claim_id="1", claim="Identical Claim", label="SUPPORT", source_dataset="test", evidence_text="Ev1")
    r2 = UnifiedClaimRecord(claim_id="2", claim="Identical Claim", label="SUPPORT", source_dataset="test", evidence_text="Ev1")
    r3 = UnifiedClaimRecord(claim_id="3", claim="Unique Claim", label="REFUTE", source_dataset="test", evidence_text="Ev2")

    records = [r1, r2, r3]
    deduped, dup_count = detect_duplicates(records)

    assert len(deduped) == 2
    assert dup_count == 1
    assert deduped[0].claim_id == "1"
    assert deduped[1].claim_id == "3"


def test_preprocess_records():
    r1 = UnifiedClaimRecord(claim_id="1", claim="<p>Claim 1</p>", label="SUPPORT", source_dataset="test")
    records, stats = preprocess_records([r1])

    assert records[0].claim == "Claim 1"
    assert stats["initial_count"] == 1
    assert stats["duplicates_removed"] == 0
