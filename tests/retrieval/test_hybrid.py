"""Unit tests for src/retrieval/hybrid.py (RRF and Ablation Modes)."""

import pytest
from src.retrieval.hybrid import compute_reciprocal_rank_fusion, HybridRetriever
from src.retrieval.config import RetrievalConfig


def test_reciprocal_rank_fusion():
    sparse_res = [
        {"doc_id": "d1", "rank": 1, "score": 10.0, "dataset": "scifact", "title": "T1", "text": "Text 1"},
        {"doc_id": "d2", "rank": 2, "score": 5.0, "dataset": "scifact", "title": "T2", "text": "Text 2"}
    ]

    dense_res = [
        {"doc_id": "d2", "rank": 1, "score": 0.9, "dataset": "scifact", "title": "T2", "text": "Text 2"},
        {"doc_id": "d3", "rank": 2, "score": 0.8, "dataset": "scifact", "title": "T3", "text": "Text 3"}
    ]

    hybrid_res = compute_reciprocal_rank_fusion(sparse_res, dense_res, rrf_k=60)

    assert len(hybrid_res) == 3
    # d2 appears in both rankings (Rank 2 in sparse, Rank 1 in dense), so it should have highest RRF score!
    assert hybrid_res[0]["doc_id"] == "d2"
    assert "bm25_rank" in hybrid_res[0]
    assert "dense_rank" in hybrid_res[0]
    assert "score" in hybrid_res[0]
    assert "rrf_score" in hybrid_res[0]
    assert hybrid_res[0]["score"] > 0.0
    assert hybrid_res[0]["score"] == hybrid_res[0]["rrf_score"]


def test_hybrid_retriever_ablation_modes(tmp_path):
    config = RetrievalConfig(
        bm25_index_file=str(tmp_path / "bm25.pkl"),
        faiss_index_file=str(tmp_path / "faiss.bin"),
        doc_id_map_file=str(tmp_path / "map.json")
    )
    retriever = HybridRetriever(config=config)

    # Test unknown mode raises ValueError
    with pytest.raises(ValueError):
        retriever.retrieve("query", mode="invalid_mode")
