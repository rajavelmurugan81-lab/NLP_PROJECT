"""Unit and integration tests for OpenAlex paper retrieval, abstract reconstruction, deduplication, and domain-aware ranking."""

import pytest
from src.retrieval.openalex import (
    reconstruct_abstract,
    extract_authors_from_work,
    OpenAlexPaperRetriever
)
from src.retrieval.paper_source import (
    normalize_doi,
    normalize_title,
    deduplicate_papers,
    rank_papers_by_domain,
    UnifiedPaperRetriever
)


def test_abstract_reconstruction():
    """Test reconstructing plain text abstract from OpenAlex inverted index."""
    inv_index = {
        "Sleep": [0],
        "duration": [1],
        "affects": [2],
        "cognitive": [3],
        "performance.": [4]
    }
    reconstructed = reconstruct_abstract(inv_index)
    assert reconstructed == "Sleep duration affects cognitive performance."


def test_abstract_reconstruction_empty_or_malformed():
    """Test abstract reconstruction with null/empty/malformed inverted index."""
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""
    assert reconstruct_abstract({"test": [-1]}) == ""
    assert reconstruct_abstract("invalid_type") == ""


def test_extract_authors():
    """Test author extraction from OpenAlex work object."""
    work = {
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}}
        ]
    }
    authors = extract_authors_from_work(work)
    assert authors == ["Alice Smith", "Bob Jones"]


def test_normalize_work_schema():
    """Test schema normalization of OpenAlex work object."""
    retriever = OpenAlexPaperRetriever()
    work = {
        "id": "https://openalex.org/W123456789",
        "display_name": "Test Paper Title",
        "publication_year": 2021,
        "authorships": [{"author": {"display_name": "Author One"}}],
        "abstract_inverted_index": {"Abstract": [0], "text": [1]},
        "doi": "https://doi.org/10.1000/182",
        "open_access": {"is_oa": True},
        "concepts": [{"display_name": "Computer Science"}]
    }

    norm = retriever.normalize_work(work, rank=1)
    assert norm["paper_id"] == "W123456789"
    assert norm["title"] == "Test Paper Title"
    assert norm["authors"] == "Author One"
    assert norm["publication_year"] == 2021
    assert norm["abstract"] == "Abstract text"
    assert norm["doi"] == "https://doi.org/10.1000/182"
    assert norm["source"] == "openalex"
    assert norm["open_access"] is True
    assert "Computer Science" in norm["concepts"]


def test_doi_deduplication():
    """Test deduplication of papers sharing the same canonical DOI."""
    papers = [
        {
            "paper_id": "p1",
            "title": "Sleep and Memory",
            "doi": "https://doi.org/10.1016/j.sleep.2020.01",
            "source": "openalex"
        },
        {
            "paper_id": "p2",
            "title": "Sleep and Memory (Duplicate)",
            "doi": "10.1016/j.sleep.2020.01",
            "source": "local_scifact"
        }
    ]

    deduped = deduplicate_papers(papers)
    assert len(deduped) == 1
    assert deduped[0]["paper_id"] == "p1"


def test_title_deduplication():
    """Test deduplication of papers sharing identical titles without DOIs."""
    papers = [
        {
            "paper_id": "p1",
            "title": "CRISPR-Cas9 Genome Editing in Human Cells!",
            "doi": None,
            "source": "openalex"
        },
        {
            "paper_id": "p2",
            "title": "crispr cas9 genome editing in human cells",
            "doi": None,
            "source": "local_sciclaimhunt"
        }
    ]

    deduped = deduplicate_papers(papers)
    assert len(deduped) == 1
    assert deduped[0]["paper_id"] == "p1"


def test_domain_aware_ranking_cs_query():
    """Test domain-aware ranking boosting CS/AI papers on LLM queries."""
    query = "How does model size affect the accuracy of large language models?"
    papers = [
        {
            "paper_id": "bio_01",
            "title": "Insomnia treatment in older adults",
            "abstract": "Clinical trial of insomnia medication.",
            "score": 0.90
        },
        {
            "paper_id": "cs_01",
            "title": "Scaling Laws for Neural Language Models",
            "abstract": "We study empirical scaling laws for language model parameter size and compute.",
            "score": 0.80
        }
    ]

    ranked = rank_papers_by_domain(query, papers)
    assert ranked[0]["paper_id"] == "cs_01"
    assert ranked[0]["rank"] == 1


def test_api_failure_handling(monkeypatch):
    """Test graceful fallback returning empty list when OpenAlex API raises an exception."""
    retriever = OpenAlexPaperRetriever()

    def mock_urlopen_error(*args, **kwargs):
        raise RuntimeError("Simulated network timeout/failure")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_error)
    results = retriever.search("non_cached_test_query_12345", top_k=5)
    assert results == []


def test_empty_search_query():
    """Test searching with empty string returns empty list."""
    retriever = OpenAlexPaperRetriever()
    assert retriever.search("", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []


def test_unified_paper_retriever():
    """Test UnifiedPaperRetriever combining sources cleanly."""
    unified = UnifiedPaperRetriever()
    papers = unified.search("sleep duration cognitive performance", top_k=5)
    assert isinstance(papers, list)
    assert len(papers) > 0
    assert any(p["source"] in ["openalex", "local_scifact", "local_sciclaimhunt"] for p in papers)
