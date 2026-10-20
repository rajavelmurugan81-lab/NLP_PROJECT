"""Unit tests for src/data/schema.py."""

import pytest
from src.data.schema import UnifiedClaimRecord, CorpusDocumentRecord


def test_unified_claim_record_creation():
    rec = UnifiedClaimRecord(
        claim_id="test_001",
        claim="CRISPR-Cas9 enables precise gene editing in human cells.",
        label="SUPPORT",
        source_dataset="scifact",
        split="train",
        evidence_id="doc_101",
        evidence_text="In human cell cultures, CRISPR-Cas9 targeted specific loci with 95% efficiency.",
        evidence_title="Gene Editing Efficiency Study",
        metadata={"p_value": 0.001}
    )

    assert rec.claim_id == "test_001"
    assert rec.label == "SUPPORT"
    assert rec.source_dataset == "scifact"
    assert rec.metadata["p_value"] == 0.001


def test_unified_claim_record_to_dict_from_dict():
    rec = UnifiedClaimRecord(
        claim_id="test_002",
        claim="Vitamin C prevents all viral infections.",
        label="REFUTE",
        source_dataset="covert",
        split="dev"
    )

    d = rec.to_dict()
    assert isinstance(d, dict)
    assert d["claim_id"] == "test_002"
    assert d["label"] == "REFUTE"

    reconstructed = UnifiedClaimRecord.from_dict(d)
    assert reconstructed.claim_id == rec.claim_id
    assert reconstructed.claim == rec.claim
    assert reconstructed.label == rec.label
    assert reconstructed.source_dataset == rec.source_dataset
    assert reconstructed.split == rec.split


def test_corpus_document_record():
    doc = CorpusDocumentRecord(
        doc_id="doc_500",
        title="Sample Abstract Title",
        abstract=["Sentence 1.", "Sentence 2."],
        source_dataset="scifact",
        structured=False
    )

    d = doc.to_dict()
    assert d["doc_id"] == "doc_500"
    assert len(d["abstract"]) == 2

    recon = CorpusDocumentRecord.from_dict(d)
    assert recon.doc_id == doc.doc_id
    assert recon.abstract == doc.abstract
