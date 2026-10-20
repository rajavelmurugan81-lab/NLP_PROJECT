"""Hybrid Retriever module combining Sparse (BM25), Dense (FAISS), and OpenAlex via Reciprocal Rank Fusion (RRF)."""

from typing import List, Dict, Any, Optional, Tuple
from src.retrieval.config import RetrievalConfig, DEFAULT_RETRIEVAL_CONFIG
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.openalex import OpenAlexPaperRetriever
from src.retrieval.paper_evidence import check_paper_domain_relevance, get_domain_matched_papers
from src.retrieval.paper_source import deduplicate_papers, rank_papers_by_domain


def compute_reciprocal_rank_fusion(
    sparse_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across sparse and dense rankings.

    RRF_score(d) = sum(1 / (k + rank(d)))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    sparse_ranks: Dict[str, Tuple[Optional[int], float]] = {}
    dense_ranks: Dict[str, Tuple[Optional[int], float]] = {}

    for item in sparse_results:
        doc_id = item["doc_id"]
        rank = item["rank"]
        score = item.get("score", 0.0)
        sparse_ranks[doc_id] = (rank, score)
        doc_map[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    for item in dense_results:
        doc_id = item["doc_id"]
        rank = item["rank"]
        score = item.get("score", 0.0)
        dense_ranks[doc_id] = (rank, score)
        doc_map[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    hybrid_results = []
    for final_rank, (doc_id, rrf_score) in enumerate(sorted_docs, 1):
        item = doc_map[doc_id]
        s_rank, s_score = sparse_ranks.get(doc_id, (None, 0.0))
        d_rank, d_score = dense_ranks.get(doc_id, (None, 0.0))
        dataset_name = item.get("dataset") or item.get("source") or "scifact"

        hybrid_results.append({
            "doc_id": doc_id,
            "paper_id": item.get("paper_id", doc_id),
            "rank": final_rank,
            "score": round(rrf_score, 6),
            "rrf_score": round(rrf_score, 6),
            "bm25_rank": s_rank,
            "bm25_score": s_score,
            "dense_rank": d_rank,
            "dense_score": d_score,
            "dataset": dataset_name,
            "source": item.get("source", dataset_name),
            "title": item.get("title", ""),
            "text": item.get("text") or item.get("abstract", ""),
            "abstract": item.get("abstract") or item.get("text", ""),
            "url_doi": item.get("url_doi") or item.get("doi", ""),
            "doi": item.get("doi") or item.get("url_doi", ""),
            "authors": item.get("authors", "Scientific Authors"),
            "publication_year": item.get("publication_year", 2022),
            "metadata": item.get("metadata", {})
        })

    return hybrid_results


class HybridRetriever:
    """Hybrid Evidence Retriever supporting Sparse (BM25), Dense (FAISS), OpenAlex, and Hybrid (RRF) modes."""

    def __init__(
        self,
        sparse_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        openalex_retriever: Optional[OpenAlexPaperRetriever] = None,
        config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG
    ):
        self.config = config
        self.sparse_retriever = sparse_retriever or BM25Retriever(config=config)
        self.dense_retriever = dense_retriever or DenseRetriever(config=config)
        self.openalex_retriever = openalex_retriever or OpenAlexPaperRetriever()

    def load_indexes(self) -> bool:
        """Load pre-built BM25 and FAISS indexes from disk."""
        sparse_ok = self.sparse_retriever.load_index()
        dense_ok = self.dense_retriever.load_index()
        return sparse_ok and dense_ok

    def retrieve(
        self,
        query: str,
        mode: str = "openalex_hybrid",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence items supporting 5 retrieval modes."""
        k = top_k or self.config.top_k_final
        mode_lower = mode.lower().strip()

        if mode_lower in ("sparse", "bm25"):
            raw_results = self.sparse_retriever.retrieve(query, top_k=k)
        elif mode_lower == "dense":
            raw_results = self.dense_retriever.retrieve(query, top_k=k)
        elif mode_lower == "openalex":
            raw_results = self.openalex_retriever.search(query, top_k=k)
        elif mode_lower in ("local_hybrid", "bm25_faiss_rrf"):
            sparse_res = self.sparse_retriever.retrieve(query, top_k=self.config.top_k_sparse)
            dense_res = self.dense_retriever.retrieve(query, top_k=self.config.top_k_dense)
            raw_results = compute_reciprocal_rank_fusion(
                sparse_results=sparse_res,
                dense_results=dense_res,
                rrf_k=self.config.rrf_k
            )
        elif mode_lower in ("openalex_hybrid", "hybrid", "default"):
            sparse_res = self.sparse_retriever.retrieve(query, top_k=self.config.top_k_sparse)
            dense_res = self.dense_retriever.retrieve(query, top_k=self.config.top_k_dense)
            openalex_res = self.openalex_retriever.search(query, top_k=k)

            combined_sparse = sparse_res + openalex_res
            combined_dense = dense_res + openalex_res
            
            raw_results = compute_reciprocal_rank_fusion(
                sparse_results=combined_sparse,
                dense_results=combined_dense,
                rrf_k=self.config.rrf_k
            )
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Supported modes: openalex_hybrid, openalex, local_hybrid, sparse, dense.")

        # Deduplicate papers by DOI and title
        unique_results = deduplicate_papers(raw_results)

        # Domain relevance filtering to eliminate unrelated passages
        filtered_results = []
        for doc in unique_results:
            is_relevant, _ = check_paper_domain_relevance(
                query=query,
                paper_text=doc.get("text", "") or doc.get("abstract", ""),
                paper_title=doc.get("title", "")
            )
            if is_relevant:
                filtered_results.append(doc)

        if filtered_results:
            ranked = rank_papers_by_domain(query, filtered_results)
            return ranked[:k]

        # Domain-matched paper retrieval fallback for out-of-corpus queries
        domain_matched = get_domain_matched_papers(query=query, top_k=k)
        if domain_matched:
            return domain_matched

        # If no papers match domain and no domain fallback exists, return empty
        return []
