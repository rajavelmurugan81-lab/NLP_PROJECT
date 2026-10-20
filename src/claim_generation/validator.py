"""Validation and deduplication utilities for generated scientific claims."""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional

from src.claim_generation.schema import ResearchQuestion, GeneratedClaim

logger = logging.getLogger(__name__)

# Patterns used to detect vague assertions or fabricated citation references in LLM text
FABRICATED_CITATION_PATTERNS = [
    r"\bet al\.",
    r"\([^\)]*\d{4}\)",
    r"\[\d+\]",
    r"doi:\s*10\.\d{4}/",
    r"PMID:\s*\d+"
]

VAGUE_TERMS = [
    "good for", "bad for", "is important", "helps things", "stuff", "some people say"
]

GENERIC_TEMPLATE_PATTERNS = [
    r"empirical evidence demonstrates a direct quantitative relationship involving",
    r"modulation of key pathways associated with",
    r"comparative clinical studies indicate that variations in",
    r"target biological process"
]

BIOMEDICAL_SPECIFIC_TERMS = [
    "key pathways",
    "downstream functional responses",
    "physiological outcomes",
    "clinical studies",
    "cell apoptosis",
    "tumor suppressor",
    "caspase-3",
    "neoplastic tissues",
    "ovarian cancer",
    "prefrontal cortex"
]

QUESTION_STOP_WORDS = {
    "what", "how", "does", "which", "is", "are", "do", "can", "should",
    "the", "a", "an", "in", "on", "of", "for", "to", "with", "between",
    "relationship", "effect", "impact", "influence", "affect", "role", "play"
}


def validate_research_question(question: Any) -> Tuple[bool, str, str]:
    """Validate user research question input. Returns (is_valid, normalized_text, error_msg)."""
    if question is None:
        return False, "", "Research question cannot be None."

    text = str(question).strip()
    if not text:
        return False, "", "Research question cannot be empty."

    if len(text) < 10:
        return False, text, f"Research question too short ({len(text)} chars). Must be at least 10 characters."

    if len(text) > 1000:
        return False, text[:1000], f"Research question exceeds maximum length of 1000 characters."

    # Remove extra spaces
    normalized = re.sub(r"\s+", " ", text)
    return True, normalized, ""


def validate_generated_claim(
    claim: GeneratedClaim,
    research_question: Optional[str] = None
) -> Tuple[bool, str]:
    """Validate an individual generated claim. Returns (is_valid, error_reason)."""
    if not claim or not claim.claim_text:
        return False, "Empty claim text."

    text = claim.claim_text.strip()
    if len(text) < 15:
        return False, f"Claim text too short ({len(text)} chars). Must be at least 15 characters for scientific specificity."

    lower_text = text.lower()

    # Check for vague language
    for vague in VAGUE_TERMS:
        if vague in lower_text:
            return False, f"Claim contains vague term '{vague}'."

    # Check for fabricated citations
    for pat in FABRICATED_CITATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False, "Claim contains explicit citation/ref pattern which must not be hardcoded in generated text."

    # Check for generic ungrounded template patterns
    for pat in GENERIC_TEMPLATE_PATTERNS:
        if re.search(pat, lower_text):
            return False, f"Claim uses generic ungrounded template phrase matching '{pat}'."

    # Check for grammatical malformations (e.g. 'predict educational', 'affect the accuracy and')
    malformed_patterns = [
        r"\b(predict|affect|impact|correlate)\s+(educational|cognitive|clinical|computational)\s+(significantly|directly|alters)\b",
        r"\b(predict|affect)\s+(educational|computational)\b(?!\s+(outcomes|performance|cost|metrics|models|results|data))"
    ]
    for pat in malformed_patterns:
        if re.search(pat, lower_text):
            return False, "Claim contains semantically malformed phrase structure."

    # Question-grounding and domain overlap validation if research_question provided
    if research_question:
        q_lower = research_question.lower()

        # Check domain pollution: Bio terms in non-bio questions
        bio_keywords = ["dna", "rna", "mirna", "gene", "cancer", "cell", "protein", "sleep", "disease", "patient", "clinical", "cortex", "tumor", "tissue", "apoptosis", "pathway", "biological", "caspase", "oncogene", "ovarian", "carcinoma"]
        is_question_bio = any(k in q_lower for k in bio_keywords)

        if not is_question_bio:
            for term in BIOMEDICAL_SPECIFIC_TERMS:
                if term in lower_text:
                    return False, f"Claim introduces unrelated biomedical terminology ('{term}') for a non-biological research question."

        # Semantic concept overlap check
        q_words = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", q_lower)
            if w not in QUESTION_STOP_WORDS
        ]
        if q_words:
            matched_words = [w for w in q_words if w in lower_text]
            if not matched_words:
                return False, "Claim shares zero semantic concept overlap with the research question."

    return True, ""


def deduplicate_claims(claims: List[GeneratedClaim]) -> List[GeneratedClaim]:
    """Deduplicate claims based on normalized text content."""
    seen_texts = set()
    unique_claims = []

    for claim in claims:
        norm_text = re.sub(r"[^\w\s]", "", claim.claim_text.lower().strip())
        if norm_text and norm_text not in seen_texts:
            seen_texts.add(norm_text)
            unique_claims.append(claim)
        else:
            logger.info(f"Filtered duplicate claim: '{claim.claim_text}'")

    return unique_claims
