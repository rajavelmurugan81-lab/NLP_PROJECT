"""Passage Extractor module for CiteGuard.

Splits retrieved scientific paper abstracts and texts into sentence-level passage evidence
with deterministic passage IDs while preserving exact verbatim text without paraphrasing.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def clean_sentence(text: str) -> str:
    """Strip extra whitespace while preserving exact original wording and punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def split_text_into_sentences(text: str) -> List[str]:
    """Split abstract or paper text into individual sentence passages.
    
    Uses sentence boundary pattern preserving citations, abbreviations, and numbers.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned_text = clean_sentence(text)
    if not cleaned_text:
        return []

    # Regex splitting on sentence terminal punctuation followed by space and uppercase letter
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned_text)
    
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) >= 15: # Ignore trivial fragments under 15 chars
            sentences.append(s_clean)

    if not sentences and cleaned_text:
        sentences = [cleaned_text]

    return sentences


class PassageExtractor:
    """Extracts sentence-level evidence passages with deterministic IDs from paper records."""

    def extract_passages_from_paper(self, paper: Dict[str, Any], paper_index: int = 1) -> List[Dict[str, Any]]:
        """Extract deterministic passage evidence objects from a single paper record."""
        raw_paper_id = str(paper.get("paper_id") or paper.get("doc_id") or f"P{paper_index:03d}")
        
        # Clean paper_id for deterministic passage ID format: P001, P002, etc.
        if raw_paper_id.isdigit():
            paper_id_code = f"P{int(raw_paper_id):03d}" if int(raw_paper_id) < 1000 else f"P{raw_paper_id}"
        elif raw_paper_id.startswith("W") and raw_paper_id[1:].isdigit():
            paper_id_code = f"P{paper_index:03d}"
        elif not raw_paper_id.startswith("P"):
            paper_id_code = f"P{paper_index:03d}"
        else:
            paper_id_code = raw_paper_id

        title = str(paper.get("title", "Untitled Document"))
        authors = paper.get("authors", "Scientific Authors")
        if isinstance(authors, list):
            authors = ", ".join(authors)
        
        year = paper.get("publication_year", 2022)
        doi = str(paper.get("doi") or paper.get("url_doi") or "")
        source = str(paper.get("source", "openalex"))
        rel_score = float(paper.get("relevance_score") or paper.get("score") or 0.85)

        abstract_text = str(paper.get("abstract") or paper.get("text") or "")
        sentences = split_text_into_sentences(abstract_text)

        is_eligible = paper.get("evidence_eligible", True)
        is_topic_match = paper.get("topic_match", True)

        passages = []
        for p_idx, sentence in enumerate(sentences, 1):
            passage_id = f"{paper_id_code}_P{p_idx:02d}"
            
            if not is_eligible or not is_topic_match or rel_score <= 0.0:
                p_score = 0.0
                passage_eligible = False
            else:
                p_score = round(max(0.40, rel_score - (p_idx - 1) * 0.01), 4)
                passage_eligible = True

            passages.append({
                "paper_id": raw_paper_id,
                "paper_id_code": paper_id_code,
                "passage_id": passage_id,
                "text": sentence,
                "exact_passage": sentence,
                "source": source,
                "title": title,
                "authors": authors,
                "publication_year": year,
                "doi": doi,
                "url_doi": doi,
                "relevance_score": p_score,
                "score": p_score,
                "passage_index": p_idx,
                "evidence_eligible": passage_eligible,
                "topic_match": is_topic_match
            })

        return passages

    def extract_passages_from_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract passages from a list of retrieved paper records."""
        all_passages = []
        for idx, paper in enumerate(papers, 1):
            paper_passages = self.extract_passages_from_paper(paper, paper_index=idx)
            all_passages.extend(paper_passages)
        return all_passages
