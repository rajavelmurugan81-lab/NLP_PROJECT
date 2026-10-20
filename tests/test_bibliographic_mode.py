"""Tests for Bibliographic Search Mode in CiteGuard."""

import pytest
from src.claim_generation.generator import ClaimGenerator
from src.claim_generation.classifier import QuestionType


def test_bibliographic_mode_no_causal_claims():
    generator = ClaimGenerator(provider_type="mock")
    query = "what are the 10 most cited papers published in 2026 on quantum resistant federated learning and provide their exact DOI,citation count and experimental results?"
    
    rq = generator.generate_claims(query)
    
    assert rq.metadata.get("question_type") == QuestionType.BIBLIOGRAPHIC_SEARCH.value
    assert len(rq.generated_claims) == 3
    
    # Assert none of the generated claims contain malformed causal templates
    for claim in rq.generated_claims:
        c_lower = claim.claim_text.lower()
        assert "increasing the 10 most cited" not in c_lower
        assert "directly affect deprivation" not in c_lower
        assert "systematic changes to" not in c_lower
        # Assert claims represent bibliographic assertions
        assert any(w in c_lower for w in ["literature", "doi", "citation", "experimental", "finding", "publication"])


def test_bibliographic_mode_subquestions():
    generator = ClaimGenerator(provider_type="mock")
    query = "what are the 10 most cited papers published in 2026 on quantum resistant federated learning and provide their exact DOI,citation count and experimental results?"
    
    sub_qs = generator.decompose_question(query)
    assert len(sub_qs) >= 2
    for sq in sub_qs:
        assert "quantum resistant federated learning" in sq.lower()
        assert "peer-reviewed" in sq.lower() or "doi" in sq.lower() or "experimental" in sq.lower()
