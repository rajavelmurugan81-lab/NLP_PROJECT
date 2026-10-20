"""Evidence Mapper & Relationship Classifier for CiteGuard.

Maps generated atomic scientific claims to extracted paper passages,
classifies evidence relationships (SUPPORTS, CONTRADICTS, PARTIALLY_SUPPORTS, IRRELEVANT),
evaluates evidence sufficiency, and constructs structured ClaimEvidenceLink graphs.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Strict evidence relevance and support thresholds
HIGH_RELEVANCE_THRESHOLD = 0.80
SUPPORT_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.50


@dataclass
class ClaimEvidenceLink:
    """Dataclass representing an explicit claim-to-passage evidence link with provenance."""
    claim_id: str
    paper_id: str
    passage_id: str
    relationship: str  # SUPPORTS, CONTRADICTS, PARTIALLY_SUPPORTS, IRRELEVANT
    evidence_text: str
    paper_title: str
    doi: str
    source: str
    retrieval_score: float
    evidence_score: float
    confidence: float
    authors: str = "OpenAlex Scientific Authors"
    publication_year: int = 2022

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_stem(word: str) -> str:
    """Simple rule-based suffix stemmer for scientific matching."""
    w = word.lower().strip()
    for suffix in ["ing", "ed", "es", "s", "ion", "ive", "al", "ly"]:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def tokenize_words(text: str) -> List[str]:
    """Extract lowercase stemmed word tokens filtering common English and academic meta stop words."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "been", "have", "has", "had", "does", "did", "doing", "can", "could", "should",
        "would", "may", "might", "must", "shall", "about", "above", "across", "after",
        "directly", "affects", "correlate", "measurable", "changes", "variations",
        "peer", "reviewed", "scientific", "literature", "documents", "indexed", "research",
        "published", "sources", "authoritative", "study", "studies", "synthesised", "synthesized",
        "show", "shows", "shown", "dataset", "datasets", "result", "results", "paper", "papers",
        "work", "state", "well", "achieve", "achieves", "achieved", "also", "using", "used"
    }
    return [get_stem(w) for w in words if w not in stop_words]


def classify_relationship(claim_text: str, passage_text: str) -> Tuple[str, float]:
    """Classify the evidence relationship between a claim and an extracted passage.
    
    Returns:
        Tuple[str, float]: (relationship_label, confidence_score)
        where relationship_label is one of: SUPPORTS, CONTRADICTS, PARTIALLY_SUPPORTS, IRRELEVANT
    """
    if not claim_text or not passage_text:
        return "IRRELEVANT", 0.0

    p_lower = passage_text.lower()
    c_lower = claim_text.lower()

    if "supporting the claim" in p_lower or "supports the claim" in p_lower or "supports claim" in p_lower:
        return "SUPPORTS", 0.90
    if "contradicting the claim" in p_lower or "contradicts the claim" in p_lower or "refutes the claim" in p_lower:
        return "CONTRADICTS", 0.90

    c_words = set(tokenize_words(claim_text))
    p_words = set(tokenize_words(passage_text))

    if not c_words or not p_words:
        return "IRRELEVANT", 0.0

    overlap = c_words.intersection(p_words)
    overlap_ratio = len(overlap) / float(len(c_words))

    # Low semantic overlap or fewer than 2 matching non-stop concepts means irrelevant
    if len(overlap) < 2 and overlap_ratio < 0.20:
        return "IRRELEVANT", 0.15

    p_lower = passage_text.lower()
    c_lower = claim_text.lower()

    # Contradiction indicators
    contradict_signals = [
        "no effect", "no significant", "does not", "did not", "failed to",
        "opposite", "contrary", "in contrast", "refute", "contradict",
        "inverse", "negative correlation", "no association", "no relationship",
        "no impact", "unrelated"
    ]

    # Directional inversion indicators (e.g. claim asserts reduction while evidence asserts increase)
    c_decrease = any(w in c_lower for w in ["reduces", "decreases", "lowers", "suppresses", "inhibits", "diminishes"])
    c_increase = any(w in c_lower for w in ["increases", "elevates", "promotes", "stimulates", "amplifies", "enhances", "causes higher"])
    p_decrease = any(w in p_lower for w in ["reduces", "decreases", "lowers", "suppresses", "inhibits", "diminishes"])
    p_increase = any(w in p_lower for w in ["increases", "elevates", "promotes", "stimulates", "amplifies", "enhances", "causes higher"])

    if (c_decrease and p_increase) or (c_increase and p_decrease):
        rel_score = round(min(0.95, 0.65 + overlap_ratio * 0.35), 4)
        return "CONTRADICTS", rel_score

    # Partial support indicators
    partial_signals = [
        "partially", "some", "modest", "conditional", "under certain",
        "mixed results", "inconsistent", "preliminary", "suggests possible"
    ]

    has_contradict_signal = any(sig in p_lower for sig in contradict_signals)
    has_partial_signal = any(sig in p_lower for sig in partial_signals)

    # If claim asserts a positive/direct effect but passage asserts no/inverse effect
    if has_contradict_signal:
        if any(w in c_lower for w in ["improves", "increases", "reduces", "causes", "leads", "associates", "correlates", "directly", "prevent", "promotes"]):
            rel_score = round(min(0.95, 0.60 + overlap_ratio * 0.40), 4)
            return "CONTRADICTS", rel_score

    if has_partial_signal:
        rel_score = round(min(0.90, 0.55 + overlap_ratio * 0.45), 4)
        return "PARTIALLY_SUPPORTS", rel_score

    # High semantic overlap without contradiction signals -> SUPPORTS
    if overlap_ratio >= 0.25 or len(overlap) >= 3:
        rel_score = round(min(0.98, 0.70 + overlap_ratio * 0.30), 4)
        return "SUPPORTS", rel_score

    if overlap_ratio >= 0.18 or len(overlap) >= 2:
        return "PARTIALLY_SUPPORTS", round(0.60 + overlap_ratio * 0.25, 4)

    return "IRRELEVANT", 0.30


class EvidenceMapper:
    """Maps scientific claims to sentence evidence passages and evaluates relationship & sufficiency."""

    def map_claim_to_passages(
        self,
        claim_id: str,
        claim_text: str,
        passages: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[ClaimEvidenceLink]:
        """Map an atomic claim against candidate passages and generate structured links."""
        links: List[ClaimEvidenceLink] = []

        for passage in passages:
            if passage.get("evidence_eligible") is False or float(passage.get("relevance_score") or passage.get("score") or 1.0) <= 0.0:
                continue

            p_text = passage.get("text") or passage.get("exact_passage", "")
            rel_label, ev_score = classify_relationship(claim_text, p_text)

            if rel_label == "IRRELEVANT":
                continue

            r_score = float(passage.get("relevance_score") or passage.get("score") or 0.80)
            combined_score = round((r_score * 0.4) + (ev_score * 0.6), 4)

            links.append(
                ClaimEvidenceLink(
                    claim_id=claim_id,
                    paper_id=str(passage.get("paper_id", "")),
                    passage_id=str(passage.get("passage_id", "")),
                    relationship=rel_label,
                    evidence_text=p_text,
                    paper_title=str(passage.get("title", "")),
                    doi=str(passage.get("doi") or passage.get("url_doi", "")),
                    source=str(passage.get("source", "openalex")),
                    retrieval_score=r_score,
                    evidence_score=ev_score,
                    confidence=combined_score,
                    authors=str(passage.get("authors", "OpenAlex Scientific Authors")),
                    publication_year=int(passage.get("publication_year") or passage.get("year") or 2022)
                )
            )

        # Sort links by combined evidence score
        links.sort(key=lambda x: x.evidence_score, reverse=True)
        return links[:top_k]

    def evaluate_sufficiency(
        self,
        links: List[ClaimEvidenceLink],
        min_passages: int = 1,
        min_score: float = 0.50
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate whether retrieved evidence links provide sufficient support for claim verification."""
        relevant_links = [l for l in links if l.relationship in ["SUPPORTS", "CONTRADICTS", "PARTIALLY_SUPPORTS"]]
        
        supporting_count = sum(1 for l in relevant_links if l.relationship == "SUPPORTS")
        contradicting_count = sum(1 for l in relevant_links if l.relationship == "CONTRADICTS")
        partial_count = sum(1 for l in relevant_links if l.relationship == "PARTIALLY_SUPPORTS")

        strongest_score = max((l.evidence_score for l in relevant_links), default=0.0)
        avg_score = (
            round(sum(l.evidence_score for l in relevant_links) / float(len(relevant_links)), 4)
            if relevant_links else 0.0
        )

        metrics = {
            "total_passages": len(relevant_links),
            "supporting_count": supporting_count,
            "contradicting_count": contradicting_count,
            "partial_count": partial_count,
            "strongest_score": strongest_score,
            "average_score": avg_score
        }

        is_sufficient = (len(relevant_links) >= min_passages) and (strongest_score >= min_score)
        return is_sufficient, metrics
