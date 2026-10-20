"""Configuration dataclass and settings for Hybrid Retrieval Engine."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalConfig:
    """Central configuration for sparse (BM25), dense (FAISS), and hybrid (RRF) retrieval."""
    
    # Embedding & Dense model configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 64
    device: str = "cpu"  # 'cuda' if GPU available, else 'cpu'

    # Top-K parameters for multi-stage retrieval
    top_k_sparse: int = 50
    top_k_dense: int = 50
    top_k_final: int = 10

    # Reciprocal Rank Fusion (RRF) parameter
    rrf_k: int = 60

    # Directory and index persistence paths
    index_dir: str = "models/indexes"
    corpus_file: str = "data/processed/corpus_unified.jsonl"
    bm25_index_file: str = "models/indexes/bm25_index.pkl"
    faiss_index_file: str = "models/indexes/faiss_index.bin"
    doc_id_map_file: str = "models/indexes/doc_id_map.json"

    def __post_init__(self):
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.corpus_file), exist_ok=True)


# Default global instance
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()
