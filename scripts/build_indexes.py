"""Script to build and persist BM25 and FAISS dense vector indexes."""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.config import RetrievalConfig
from src.retrieval.corpus import CorpusBuilder
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever


def estimate_disk_usage(num_documents: int, embedding_dim: int) -> Dict[str, str]:
    """Estimate storage requirements for BM25 and FAISS dense indexes (Task 12)."""
    # FAISS float32 vectors = num_docs * dim * 4 bytes
    vector_bytes = num_documents * embedding_dim * 4
    # BM25 pickle estimate ~ 2KB per document
    sparse_bytes = num_documents * 2000

    return {
        "estimated_faiss_size_mb": f"{round(vector_bytes / (1024**2), 2)} MB",
        "estimated_bm25_size_mb": f"{round(sparse_bytes / (1024**2), 2)} MB",
        "total_estimated_size_mb": f"{round((vector_bytes + sparse_bytes) / (1024**2), 2)} MB"
    }


def main():
    parser = argparse.ArgumentParser(description="Build and save BM25 and FAISS retrieval indexes.")
    parser.add_argument("--max-sch-records", type=int, default=5000, help="Cap SciClaimHunt records for indexing limit.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild existing indexes.")
    args = parser.parse_args()

    config = RetrievalConfig()
    print("=== Phase 4: Building Hybrid Retrieval Indexes ===")
    print(f"Index Directory: {config.index_dir}")
    print(f"Embedding Model: {config.embedding_model_name} ({config.embedding_dimension}-dim)")

    # 1. Build Corpus
    builder = CorpusBuilder(config=config)
    max_records = {"sciclaimhunt": args.max_sch_records} if args.max_sch_records else None
    
    print("\n[Step 1/4] Assembling search corpus...")
    corpus, corpus_stats = builder.build_corpus_from_datasets(max_records_per_dataset=max_records)
    corpus_path = builder.save_corpus(corpus)

    print(f"Retained Corpus Documents: {len(corpus):,}")
    print(f"Corpus Stats: {corpus_stats}")

    # Estimate Storage
    est_storage = estimate_disk_usage(len(corpus), config.embedding_dimension)
    print(f"\n[Task 12 Disk Safety Check] Estimated Index Storage Requirements:")
    for k, v in est_storage.items():
        print(f"  - {k}: {v}")

    # 2. Build BM25 Index
    print("\n[Step 2/4] Building BM25 Sparse Index...")
    start_sparse = time.time()
    sparse_retriever = BM25Retriever(config=config)
    sparse_retriever.build_index(corpus)
    sparse_retriever.save_index()
    sparse_time = time.time() - start_sparse
    print(f"BM25 Index built in {sparse_time:.2f} seconds.")

    # 3. Build FAISS Dense Index
    print("\n[Step 3/4] Building FAISS Dense Vector Index...")
    start_dense = time.time()
    dense_retriever = DenseRetriever(config=config)
    dense_retriever.build_index(corpus)
    dense_retriever.save_index()
    dense_time = time.time() - start_dense
    print(f"FAISS Dense Index built in {dense_time:.2f} seconds.")

    # 4. Actual Disk Usage
    bm25_size = os.path.getsize(config.bm25_index_file) / (1024**2) if os.path.exists(config.bm25_index_file) else 0
    faiss_size = os.path.getsize(config.faiss_index_file) / (1024**2) if os.path.exists(config.faiss_index_file) else 0

    print("\n=== Indexing Summary ===")
    print(f"Corpus Size: {len(corpus):,} documents")
    print(f"BM25 Index Size: {bm25_size:.2f} MB (Time: {sparse_time:.2f}s)")
    print(f"FAISS Index Size: {faiss_size:.2f} MB (Time: {dense_time:.2f}s)")
    print(f"Total Indexing Time: {sparse_time + dense_time:.2f} seconds.")
    print("=== Index Construction Completed Successfully ===")


if __name__ == "__main__":
    main()
