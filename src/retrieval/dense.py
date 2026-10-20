"""Dense Vector Retriever module using SentenceTransformers and FAISS."""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG
from src.data.schema import CorpusDocumentRecord

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class DenseRetriever:
    """Dense Vector Retriever using SentenceTransformers embeddings and FAISS IndexFlatIP."""

    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG):
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[Any] = None
        self.documents: List[CorpusDocumentRecord] = []
        self.doc_ids: List[str] = []

    def _init_model(self):
        """Lazy load SentenceTransformer model."""
        if self.model is None:
            if not FAISS_AVAILABLE:
                raise ImportError("[DenseRetriever] sentence-transformers or faiss is not installed.")
            print(f"[DenseRetriever] Loading SentenceTransformer model: {self.config.embedding_model_name}...")
            self.model = SentenceTransformer(self.config.embedding_model_name, device=self.config.device)

    def build_index(self, documents: List[CorpusDocumentRecord]):
        """Encode documents in batches and build FAISS IndexFlatIP (cosine similarity)."""
        self._init_model()
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]

        texts = [f"{doc.title} {' '.join(doc.abstract)}" for doc in documents]
        print(f"[DenseRetriever] Encoding {len(texts):,} documents in batches of {self.config.batch_size}...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        print(f"[DenseRetriever] Creating FAISS index for {dimension}-dim embeddings...")
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        print("[DenseRetriever] FAISS index creation complete.")

    def save_index(self, index_filepath: Optional[str] = None, map_filepath: Optional[str] = None):
        """Save FAISS binary index and document map to disk."""
        target_index = index_filepath or self.config.faiss_index_file
        target_map = map_filepath or self.config.doc_id_map_file

        os.makedirs(os.path.dirname(target_index), exist_ok=True)
        faiss.write_index(self.faiss_index, target_index)

        map_data = {
            "doc_ids": self.doc_ids,
            "documents": [doc.to_dict() for doc in self.documents]
        }
        with open(target_map, "w", encoding="utf-8") as f:
            json.dump(map_data, f)

        print(f"[DenseRetriever] Saved FAISS index to {target_index} and map to {target_map}")

    def load_index(self, index_filepath: Optional[str] = None, map_filepath: Optional[str] = None) -> bool:
        """Load FAISS binary index and document map from disk."""
        target_index = index_filepath or self.config.faiss_index_file
        target_map = map_filepath or self.config.doc_id_map_file

        if not os.path.exists(target_index) or not os.path.exists(target_map):
            return False

        if not FAISS_AVAILABLE:
            raise ImportError("[DenseRetriever] sentence-transformers or faiss is not installed.")

        print(f"[DenseRetriever] Loading FAISS index from {target_index}...")
        self.faiss_index = faiss.read_index(target_index)

        with open(target_map, "r", encoding="utf-8") as f:
            map_data = json.load(f)

        self.doc_ids = map_data["doc_ids"]
        self.documents = [CorpusDocumentRecord.from_dict(d) for d in map_data["documents"]]
        print(f"[DenseRetriever] Successfully loaded FAISS index with {self.faiss_index.ntotal:,} vectors.")
        return True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top_k dense semantic vector matches for query."""
        if not self.faiss_index:
            raise ValueError("[DenseRetriever] FAISS index not built or loaded.")

        k = top_k or self.config.top_k_dense
        self._init_model()

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        scores, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append({
                "doc_id": doc.doc_id,
                "score": round(float(score), 4),
                "rank": rank,
                "dataset": doc.source_dataset,
                "title": doc.title,
                "text": " ".join(doc.abstract),
                "metadata": doc.metadata
            })

        return results
