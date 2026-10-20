"""Unit tests for src/retrieval/dense.py (FAISS Dense Retriever)."""

import pytest
from src.data.schema import CorpusDocumentRecord
from src.retrieval.dense import DenseRetriever, FAISS_AVAILABLE
from src.retrieval.config import RetrievalConfig


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS or SentenceTransformers not available")
def test_dense_retriever_build_retrieve_persistence(tmp_path):
    config = RetrievalConfig(
        faiss_index_file=str(tmp_path / "faiss.bin"),
        doc_id_map_file=str(tmp_path / "map.json"),
        batch_size=2
    )
    retriever = DenseRetriever(config=config)

    docs = [
        CorpusDocumentRecord(doc_id="d1", title="Gene Editing", abstract=["CRISPR-Cas9 enables targeted DNA modification."], source_dataset="scifact"),
        CorpusDocumentRecord(doc_id="d2", title="Vaccine Efficacy", abstract=["mRNA vaccines produce antibody responses against viral spike protein."], source_dataset="covert"),
        CorpusDocumentRecord(doc_id="d3", title="Pruning Study", abstract=["Neural network magnitude pruning reduces parameters."], source_dataset="sciclaimhunt")
    ]

    retriever.build_index(docs)
    retriever.save_index()

    # Retrieve query
    results = retriever.retrieve("CRISPR gene editing", top_k=2)
    assert len(results) == 2
    assert results[0]["doc_id"] == "d1"

    # Persistence load test
    loaded_retriever = DenseRetriever(config=config)
    assert loaded_retriever.load_index() is True
    loaded_results = loaded_retriever.retrieve("neural network pruning", top_k=1)
    assert len(loaded_results) == 1
    assert loaded_results[0]["doc_id"] == "d3"
