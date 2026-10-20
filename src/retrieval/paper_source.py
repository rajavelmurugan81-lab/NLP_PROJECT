"""Unified Paper Source Retrieval Interface, Deduplication, and Domain-Aware Ranking.

Provides a common paper retrieval abstraction uniting OpenAlex API paper discovery
and local corpus retrieval (SciFact / SciClaimHunt), along with DOI/title deduplication.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set

from src.retrieval.openalex import OpenAlexPaperRetriever
from src.retrieval.paper_evidence import check_paper_domain_relevance

logger = logging.getLogger(__name__)


def normalize_doi(doi: Optional[str]) -> str:
    """Normalize DOI string for deduplication."""
    if not doi or not isinstance(doi, str):
        return ""
    clean = doi.lower().strip()
    clean = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean)
    return clean.strip()


def normalize_title(title: Optional[str]) -> str:
    """Normalize title string for fuzzy title deduplication."""
    if not title or not isinstance(title, str):
        return ""
    clean = title.lower().strip()
    clean = re.sub(r"[^\w\s]", " ", clean)
    return re.sub(r"\s+", " ", clean).strip()


def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate paper records prioritizing canonical DOI match, followed by normalized title match."""
    seen_dois: Set[str] = set()
    seen_titles: Set[str] = set()
    unique_papers: List[Dict[str, Any]] = []

    for paper in papers:
        doi = normalize_doi(paper.get("doi") or paper.get("url_doi"))
        title = normalize_title(paper.get("title"))

        if doi and doi in seen_dois:
            logger.debug(f"[Deduplication] Dropping duplicate paper by DOI: {doi}")
            continue

        if title and title in seen_titles:
            logger.debug(f"[Deduplication] Dropping duplicate paper by Title: '{title}'")
            continue

        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)

        unique_papers.append(paper)

    return unique_papers


def rank_papers_by_domain(
    query: str,
    papers: List[Dict[str, Any]],
    sort_by: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Apply domain-aware relevance scoring, topic match tagging, and ranking."""
    if not papers:
        return []

    scored_papers = []
    for paper in papers:
        p_title = paper.get("title") or ""
        p_text = paper.get("abstract") or paper.get("text") or ""
        
        is_rel, domain_score = check_paper_domain_relevance(
            query=query,
            paper_text=p_text,
            paper_title=p_title
        )

        base_score = float(paper.get("score") or paper.get("relevance_score") or 0.80)
        updated_p = dict(paper)
        updated_p["topic_match"] = is_rel

        if not is_rel:
            updated_p["evidence_eligible"] = False
            updated_p["domain_score"] = 0.0
            updated_p["score"] = 0.0
            updated_p["relevance_score"] = 0.0
        else:
            updated_p["evidence_eligible"] = True
            final_score = round(max(0.50, min(0.99, (base_score + domain_score) / 2.0)), 4)
            updated_p["domain_score"] = final_score
            updated_p["score"] = final_score
            updated_p["relevance_score"] = final_score

        scored_papers.append(updated_p)

    if sort_by == "cited_by_count:desc":
        # Sort primarily by citation count descending, prioritizing topic-matched papers
        scored_papers.sort(
            key=lambda x: (
                1 if x.get("topic_match", False) else 0,
                x.get("cited_by_count", 0),
                x.get("score", 0.0)
            ),
            reverse=True
        )
    else:
        scored_papers.sort(key=lambda x: x["score"], reverse=True)
    
    # Re-assign ranks
    for idx, p in enumerate(scored_papers, 1):
        p["rank"] = idx

    return scored_papers


class BasePaperRetriever(ABC):
    """Abstract Base Class for Paper Retrievers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top_k scientific papers for given query."""
        pass


class LocalCorpusPaperRetriever(BasePaperRetriever):
    """Paper Retriever wrapping local BM25 + FAISS + RRF SciFact/SciClaimHunt index."""

    def __init__(self, hybrid_retriever: Optional[Any] = None):
        if hybrid_retriever is None:
            from src.retrieval.hybrid import HybridRetriever
            hybrid_retriever = HybridRetriever()
        self.hybrid_retriever = hybrid_retriever
        if self.hybrid_retriever.sparse_retriever.bm25_index is None:
            try:
                self.hybrid_retriever.load_indexes()
            except Exception as e:
                logger.warning(f"[LocalCorpusPaperRetriever] Index load warning: {e}")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve candidate papers from local SciFact / SciClaimHunt index."""
        try:
            raw_docs = self.hybrid_retriever.retrieve(query=query, mode="hybrid", top_k=top_k)
        except Exception as e:
            logger.warning(f"[LocalCorpusPaperRetriever] Retrieval warning: {e}")
            return []

        papers = []
        for rank, doc in enumerate(raw_docs, 1):
            dataset = doc.get("dataset", "scifact")
            source_label = f"local_{dataset}" if not dataset.startswith("local_") else dataset
            
            p_id = str(doc.get("doc_id", f"local_doc_{rank}"))
            title = str(doc.get("title", "Untitled Document"))
            text = str(doc.get("text", ""))
            
            meta = doc.get("metadata", {})
            authors = meta.get("authors", "Scientific Authors")
            year = meta.get("publication_year", 2022)
            url_doi = meta.get("url_doi", "")

            papers.append({
                "paper_id": p_id,
                "doc_id": p_id,
                "title": title,
                "authors": authors,
                "publication_year": year,
                "abstract": text,
                "text": text,
                "url_doi": url_doi,
                "doi": url_doi,
                "url": url_doi,
                "source": source_label,
                "open_access": True,
                "relevance_score": float(doc.get("score") or doc.get("rrf_score") or 0.75),
                "score": float(doc.get("score") or doc.get("rrf_score") or 0.75),
                "rank": rank
            })
        return papers


class UnifiedPaperRetriever(BasePaperRetriever):
    """Unified Paper Retriever combining OpenAlex live search and local corpus retrieval."""

    def __init__(
        self,
        openalex_retriever: Optional[OpenAlexPaperRetriever] = None,
        local_retriever: Optional[LocalCorpusPaperRetriever] = None
    ):
        self.openalex_retriever = openalex_retriever or OpenAlexPaperRetriever()
        self.local_retriever = local_retriever or LocalCorpusPaperRetriever()

    def search(
        self,
        query: str,
        top_k: int = 10,
        publication_year: Optional[int] = None,
        sort_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve real scientific papers from OpenAlex and local corpus, deduplicate, and rank."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Fetch from OpenAlex API
        openalex_papers = self.openalex_retriever.search(
            clean_query,
            top_k=top_k,
            publication_year=publication_year,
            sort_by=sort_by
        )

        # 2. Fetch from Local Corpus (if not explicitly restricted to specific modern year where local corpus has no data)
        local_papers = []
        is_future_year = publication_year and publication_year >= 2025
        if not is_future_year:
            local_papers = self.local_retriever.search(clean_query, top_k=top_k)

        # 3. Combine paper candidates
        combined = openalex_papers + local_papers

        if not combined:
            return []

        # 4. Deduplicate papers by DOI and title
        unique_papers = deduplicate_papers(combined)

        # 5. Apply domain-aware ranking
        ranked_papers = rank_papers_by_domain(clean_query, unique_papers, sort_by=sort_by)

        return ranked_papers[:top_k]
