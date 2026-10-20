"""Script to test and compare BM25, Dense, and Hybrid (RRF) evidence retrieval."""

import os
import sys
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid import HybridRetriever


def print_results(mode_name: str, results: list):
    print(f"\n========================================================")
    print(f" Mode: {mode_name.upper()} (Top {len(results)} Results)")
    print(f"========================================================")
    for res in results:
        print(f"Rank #{res['rank']} | Doc ID: {res['doc_id']} | Score/RRF: {res.get('rrf_score', res.get('score'))}")
        print(f"Dataset: {res['dataset'].upper()} | Title: {res['title']}")
        print(f"Text Snippet: {res['text'][:150]}...\n")


def main():
    parser = argparse.ArgumentParser(description="Test and compare retrieval modes.")
    parser.add_argument("--query", type=str, default="CRISPR-Cas9 enables precise gene editing in human cells", help="Claim query to search.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top evidence results to display.")
    args = parser.parse_args()

    config = RetrievalConfig()
    print("=== Testing Hybrid Retrieval Engine ===")
    print(f"Query Claim: '{args.query}'")

    retriever = HybridRetriever(config=config)
    if not retriever.load_indexes():
        print("[Error] Indexes not found. Please run 'python scripts/build_indexes.py' first.")
        return

    # 1. BM25 Sparse Mode
    bm25_results = retriever.retrieve(args.query, mode="bm25", top_k=args.top_k)
    print_results("BM25 Sparse Retrieval", bm25_results)

    # 2. Dense Mode
    dense_results = retriever.retrieve(args.query, mode="dense", top_k=args.top_k)
    print_results("Dense FAISS Retrieval", dense_results)

    # 3. Hybrid RRF Mode
    hybrid_results = retriever.retrieve(args.query, mode="hybrid", top_k=args.top_k)
    print_results("Hybrid (BM25 + Dense + RRF)", hybrid_results)


if __name__ == "__main__":
    main()
