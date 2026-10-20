"""BM25 Sparse Retriever module for scientific text evidence retrieval."""

import os
import re
import pickle
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG
from src.data.schema import CorpusDocumentRecord


def tokenize_scientific_text(text: str) -> List[str]:
    """Tokenize scientific text preserving numbers, chemical terms, and mathematical symbols."""
    if not text:
        return []
    # Tokenize words, numbers, and hyphens/underscores
    tokens = re.findall(r"\b\w+(?:[-_]\w+)*\b", text.lower())
    return tokens


class BM25Retriever:
    """Sparse Retriever implementation using BM25Okapi over scientific evidence corpus."""

    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG):
        self.config = config
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[CorpusDocumentRecord] = []
        self.doc_ids: List[str] = []

    def build_index(self, documents: List[CorpusDocumentRecord]):
        """Build BM25 index over provided corpus documents."""
        print(f"[BM25Retriever] Tokenizing and indexing {len(documents):,} documents...")
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]

        tokenized_corpus = [
            tokenize_scientific_text(f"{doc.title} {' '.join(doc.abstract)}")
            for doc in documents
        ]

        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("[BM25Retriever] BM25 indexing complete.")

    def save_index(self, filepath: Optional[str] = None):
        """Save BM25 index and document references to disk."""
        target_path = filepath or self.config.bm25_index_file
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        data = {
            "bm25_index": self.bm25_index,
            "doc_ids": self.doc_ids,
            "documents": [doc.to_dict() for doc in self.documents]
        }

        with open(target_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25Retriever] Saved BM25 index to {target_path}")

    def load_index(self, filepath: Optional[str] = None) -> bool:
        """Load pre-built BM25 index from disk."""
        target_path = filepath or self.config.bm25_index_file
        if not os.path.exists(target_path):
            return False

        print(f"[BM25Retriever] Loading BM25 index from {target_path}...")
        with open(target_path, "rb") as f:
            data = pickle.load(f)

        self.bm25_index = data["bm25_index"]
        self.doc_ids = data["doc_ids"]
        self.documents = [CorpusDocumentRecord.from_dict(d) for d in data["documents"]]
        print(f"[BM25Retriever] Successfully loaded index with {len(self.doc_ids):,} documents.")
        return True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant evidence documents for a given query."""
        if not self.bm25_index:
            raise ValueError("[BM25Retriever] Index not built or loaded.")

        k = top_k or self.config.top_k_sparse
        tokenized_query = tokenize_scientific_text(query)

        if not tokenized_query:
            return []

        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            doc = self.documents[idx]
            results.append({
                "doc_id": doc.doc_id,
                "score": round(score, 4),
                "rank": rank,
                "dataset": doc.source_dataset,
                "title": doc.title,
                "text": " ".join(doc.abstract),
                "metadata": doc.metadata
            })

        return results
