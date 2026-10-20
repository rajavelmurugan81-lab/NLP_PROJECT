"""Unit tests for src/data/validator.py."""

import pytest
from src.data.schema import UnifiedClaimRecord
from src.data.validator import DatasetValidator


def test_validator_valid_records():
    r1 = UnifiedClaimRecord(
        claim_id="valid_01",
        claim="Valid scientific claim statement.",
        label="SUPPORT",
        source_dataset="scifact",
        split="train",
        evidence_text="Valid evidence text."
    )

    validator = DatasetValidator()
    report = validator.validate([r1])

    assert report.is_valid is True
    assert report.total_records == 1
    assert report.valid_records_count == 1
    assert len(report.errors) == 0


def test_validator_empty_claim_error():
    r1 = UnifiedClaimRecord(
        claim_id="invalid_01",
        claim="   ",
        label="SUPPORT",
        source_dataset="scifact",
        split="train"
    )

    validator = DatasetValidator()
    report = validator.validate([r1])

    assert report.is_valid is False
    assert report.empty_claims_count == 1
    assert len(report.errors) > 0


def test_validator_invalid_label_error():
    r1 = UnifiedClaimRecord(
        claim_id="invalid_02",
        claim="Valid claim text",
        label="INVALID_LABEL_TAG",
        source_dataset="scifact",
        split="train"
    )

    validator = DatasetValidator()
    report = validator.validate([r1])

    assert report.is_valid is False
    assert report.invalid_label_count == 1


def test_validator_invalid_split_error():
    r1 = UnifiedClaimRecord(
        claim_id="invalid_03",
        claim="Valid claim text",
        label="SUPPORT",
        source_dataset="scifact",
        split="invalid_split_name"
    )

    validator = DatasetValidator()
    report = validator.validate([r1])

    assert report.is_valid is False
    assert report.invalid_split_count == 1


def test_validator_duplicate_id_warning():
    r1 = UnifiedClaimRecord(claim_id="dup_01", claim="Claim 1", label="SUPPORT", source_dataset="test", split="train")
    r2 = UnifiedClaimRecord(claim_id="dup_01", claim="Claim 2", label="REFUTE", source_dataset="test", split="train")

    validator = DatasetValidator()
    report = validator.validate([r1, r2])

    assert report.duplicate_id_count == 1
    assert len(report.warnings) > 0
