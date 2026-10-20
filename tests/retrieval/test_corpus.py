"""Unit tests for src/retrieval/corpus.py."""

import pytest
from src.data.schema import CorpusDocumentRecord, UnifiedClaimRecord
from src.retrieval.corpus import CorpusBuilder
from src.retrieval.config import RetrievalConfig


def test_corpus_document_record_structure():
    doc = CorpusDocumentRecord(
        doc_id="doc_test_01",
        title="Gene Editing Study",
        abstract=["CRISPR-Cas9 edits DNA."],
        source_dataset="scifact"
    )
    assert doc.doc_id == "doc_test_01"
    assert doc.title == "Gene Editing Study"
    assert doc.source_dataset == "scifact"


def test_corpus_builder_save_load(tmp_path):
    config = RetrievalConfig(
        corpus_file=str(tmp_path / "corpus_test.jsonl")
    )
    builder = CorpusBuilder(config=config)

    docs = [
        CorpusDocumentRecord(doc_id="d1", title="T1", abstract=["A1"], source_dataset="scifact"),
        CorpusDocumentRecord(doc_id="d2", title="T2", abstract=["A2"], source_dataset="covert")
    ]

    saved_path = builder.save_corpus(docs)
    loaded_docs = builder.load_corpus(saved_path)

    assert len(loaded_docs) == 2
    assert loaded_docs[0].doc_id == "d1"
    assert loaded_docs[1].doc_id == "d2"
