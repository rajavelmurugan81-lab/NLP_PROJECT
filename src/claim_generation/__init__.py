"""Claim Generation package for CiteGuard."""

from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.claim_generation.validator import (
    validate_research_question,
    validate_generated_claim,
    deduplicate_claims
)
from src.claim_generation.generator import ClaimGenerator

__all__ = [
    "ResearchQuestion",
    "GeneratedClaim",
    "validate_research_question",
    "validate_generated_claim",
    "deduplicate_claims",
    "ClaimGenerator"
]
