"""Unit tests for src/retrieval/sparse.py (BM25)."""

import pytest
from src.data.schema import CorpusDocumentRecord
from src.retrieval.sparse import BM25Retriever, tokenize_scientific_text
from src.retrieval.config import RetrievalConfig


def test_tokenize_scientific_text():
    tokens = tokenize_scientific_text("CRISPR-Cas9 edits H2O molecules with p < 0.001.")
    assert "crispr-cas9" in tokens or "crispr" in tokens
    assert "edits" in tokens


def test_bm25_retriever_build_retrieve_persistence(tmp_path):
    config = RetrievalConfig(
        bm25_index_file=str(tmp_path / "bm25.pkl")
    )
    retriever = BM25Retriever(config=config)

    docs = [
        CorpusDocumentRecord(doc_id="d1", title="Gene Editing", abstract=["CRISPR-Cas9 enables targeted DNA modification."], source_dataset="scifact"),
        CorpusDocumentRecord(doc_id="d2", title="Vaccine Efficacy", abstract=["mRNA vaccines produce antibody responses against viral spike protein."], source_dataset="covert"),
        CorpusDocumentRecord(doc_id="d3", title="Pruning Study", abstract=["Neural network magnitude pruning reduces parameters."], source_dataset="sciclaimhunt")
    ]

    retriever.build_index(docs)
    retriever.save_index()

    # Retrieve query
    results = retriever.retrieve("CRISPR gene editing", top_k=2)
    assert len(results) > 0
    assert results[0]["doc_id"] == "d1"
    assert results[0]["rank"] == 1

    # Empty query test
    assert retriever.retrieve("", top_k=2) == []

    # Persistence load test
    loaded_retriever = BM25Retriever(config=config)
    assert loaded_retriever.load_index() is True
    loaded_results = loaded_retriever.retrieve("vaccine antibody", top_k=1)
    assert len(loaded_results) == 1
    assert loaded_results[0]["doc_id"] == "d2"
