"""Question-Type Classification module for CiteGuard.

Classifies arbitrary research questions into canonical question categories:
- SCIENTIFIC_HYPOTHESIS: Empirical cause-effect or relationship queries
- COMPARATIVE_SCIENTIFIC: Comparison across systems, models, or configurations
- BIBLIOGRAPHIC_SEARCH: Queries seeking specific papers, counts, publication years, DOIs, citation rankings
- LITERATURE_REVIEW: Broad topical survey or overview queries
- METADATA_REQUEST: Specific metadata lookups (DOI, authors, journals)
"""

import re
from enum import Enum
from typing import Dict, Any, Optional


class QuestionType(str, Enum):
    """Canonical question categories."""
    SCIENTIFIC_HYPOTHESIS = "SCIENTIFIC_HYPOTHESIS"
    COMPARATIVE_SCIENTIFIC = "COMPARATIVE_SCIENTIFIC"
    BIBLIOGRAPHIC_SEARCH = "BIBLIOGRAPHIC_SEARCH"
    LITERATURE_REVIEW = "LITERATURE_REVIEW"
    METADATA_REQUEST = "METADATA_REQUEST"


def classify_question(query: str) -> Dict[str, Any]:
    """Classify user research question and extract operational parameters.
    
    Returns:
        Dict containing:
            question_type: QuestionType enum
            topic: str
            target_count: Optional[int]
            target_year: Optional[int]
            ranking_metric: str (cited_by_count, relevance, publication_date)
            requires_doi: bool
            requires_citations: bool
            requires_experimental_results: bool
    """
    if not query or not isinstance(query, str):
        return {
            "question_type": QuestionType.SCIENTIFIC_HYPOTHESIS,
            "topic": "",
            "target_count": 5,
            "target_year": None,
            "ranking_metric": "relevance",
            "requires_doi": False,
            "requires_citations": False,
            "requires_experimental_results": False
        }

    q_lower = query.strip().lower()

    # 1. Check for Bibliographic Search indicators
    biblio_patterns = [
        r"(?:what\s+are|find|list|show|retrieve|get)\s+(?:the\s+)?(?:\d+\s+)?(?:most\s+cited|top\s+cited|highest\s+cited|cited|recent|latest)?\s*papers?",
        r"(?:most\s+cited|top\s+cited|highest\s+cited)\s+papers?",
        r"published\s+in\s+\d{4}",
        r"exact\s+(?:doi|citation\s+count)",
        r"(?:provide|give)\s+(?:their|the)\s+(?:exact\s+)?(?:doi|citation\s+count)",
        r"\bdoi\b.*\bcitation\s+count\b",
        r"\btop\s+\d+\s+papers?\b"
    ]

    is_biblio = any(re.search(pat, q_lower) for pat in biblio_patterns)

    # 2. Extract Count (e.g., "10 most cited", "top 5")
    count_match = re.search(r"\b(?:the\s+)?(?P<num>\d+)\s+(?:most\s+cited|top|highest\s+cited|papers?|recent|key)\b", q_lower)
    target_count = int(count_match.group("num")) if count_match else 10

    # 3. Extract Year (e.g., "published in 2026", "2024 papers")
    year_match = re.search(r"\b(?:in|year|published\s+in|from)\s+(?P<year>20\d{2}|19\d{2})\b", q_lower)
    target_year = int(year_match.group("year")) if year_match else None

    # 4. Extract Ranking Metric
    if "most cited" in q_lower or "top cited" in q_lower or "highest cited" in q_lower or "citation count" in q_lower:
        ranking_metric = "cited_by_count"
    elif "recent" in q_lower or "latest" in q_lower:
        ranking_metric = "publication_date"
    else:
        ranking_metric = "relevance"

    requires_doi = "doi" in q_lower
    requires_citations = "citation" in q_lower
    requires_experimental_results = "experimental result" in q_lower or "experiment" in q_lower

    # 5. Extract core Topic
    clean_topic = query.strip().rstrip("?.")
    clean_topic = re.sub(
        r"^(?:what\s+are\s+the\s+\d+\s+most\s+cited\s+papers\s+published\s+in\s+\d{4}\s+on|what\s+are\s+the\s+most\s+cited\s+papers\s+on|find\s+the\s+\d+\s+papers\s+on|list\s+the\s+\d+\s+most\s+cited\s+papers\s+on|what\s+are\s+the\s+top\s+\d+\s+papers\s+on)\s+",
        "",
        clean_topic,
        flags=re.IGNORECASE
    ).strip()

    clean_topic = re.sub(
        r"\s+(?:and\s+provide\s+their\s+exact\s+doi.*|and\s+give\s+their\s+doi.*|provide\s+doi.*|with\s+exact\s+doi.*)$",
        "",
        clean_topic,
        flags=re.IGNORECASE
    ).strip()

    if is_biblio:
        q_type = QuestionType.BIBLIOGRAPHIC_SEARCH
    elif any(k in q_lower for k in ["compare", "comparison", "difference between", "versus", "vs", "trade-off", "trade off"]):
        q_type = QuestionType.COMPARATIVE_SCIENTIFIC
    elif any(k in q_lower for k in ["literature review", "overview of", "survey of", "current state of", "state of the art"]):
        q_type = QuestionType.LITERATURE_REVIEW
    elif "doi" in q_lower and ("what is the doi" in q_lower or "find doi" in q_lower):
        q_type = QuestionType.METADATA_REQUEST
    elif any(k in q_lower for k in ["how does", "how do", "by what mechanism", "in what way"]):
        # Questions asking mechanism, comparative effect, or improvement
        q_type = QuestionType.COMPARATIVE_SCIENTIFIC
    elif any(k in q_lower for k in ["model size", "parameter", "scaling", "computational cost"]):
        q_type = QuestionType.COMPARATIVE_SCIENTIFIC
    elif any(k in q_lower for k in ["what are the effects", "does ", "do ", "impact of", "role of", "relationship between", "influence"]):
        q_type = QuestionType.SCIENTIFIC_HYPOTHESIS
    else:
        q_type = QuestionType.SCIENTIFIC_HYPOTHESIS

    return {
        "question_type": q_type,
        "topic": clean_topic,
        "target_count": target_count,
        "target_year": target_year,
        "ranking_metric": ranking_metric,
        "requires_doi": requires_doi,
        "requires_citations": requires_citations,
        "requires_experimental_results": requires_experimental_results
    }
