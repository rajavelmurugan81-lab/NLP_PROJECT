"""Comprehensive test suite for CiteGuard Claim Generator and Validator across 10 distinct scientific domains."""

import pytest
from typing import List

from src.claim_generation.generator import ClaimGenerator, parse_scientific_question
from src.claim_generation.validator import validate_generated_claim, validate_research_question
from src.claim_generation.schema import GeneratedClaim, ResearchQuestion


TEST_QUESTIONS = [
    ("neuroscience", "What is the relationship between sleep duration and cognitive performance?"),
    ("cs_llm", "How does model size affect the accuracy and computational cost of large language models?"),
    ("oncology", "Does miRNA expression play a role in cancer progression?"),
    ("molecular_biology", "How does CRISPR-Cas9 enable targeted DNA cleavage?"),
    ("material_science", "What factors influence solar cell efficiency?"),
    ("cs_education", "Does machine learning predict educational outcomes?"),
    ("chemistry", "How does temperature affect catalyst activity in ammonia synthesis?"),
    ("environmental", "Does deforestation contribute to local microclimate change?"),
    ("economics", "How does inflation impact consumer spending behavior?"),
    ("plasma_physics", "What is the effect of magnetic fields on plasma confinement in tokamaks?")
]

BIOMEDICAL_TERMS = [
    "key pathways", "downstream functional responses", "physiological outcomes",
    "clinical studies", "cell apoptosis", "tumor suppressor", "caspase-3",
    "neoplastic tissues", "ovarian cancer", "prefrontal cortex"
]


def test_claim_generation_10_domains():
    """Verify claim generation across 10 distinct scientific domains."""
    generator = ClaimGenerator(provider_type="mock")
    all_generated_claims_by_question = {}

    for domain_tag, question in TEST_QUESTIONS:
        rq = generator.generate_claims(question)

        # 1. Exactly 3 atomic claims generated
        assert len(rq.generated_claims) == 3, f"Expected 3 claims for '{question}', got {len(rq.generated_claims)}"

        # 2. All claims are validated and valid
        for claim in rq.generated_claims:
            assert claim.is_valid is True
            assert claim.status == "VALIDATED"
            assert len(claim.claim_text) >= 15

            # 3. Verify semantic overlap with question
            is_v, err = validate_generated_claim(claim, research_question=question)
            assert is_v is True, f"Claim '{claim.claim_text}' failed validation for question '{question}': {err}"

        # 4. Check for domain pollution in non-biomedical questions
        if domain_tag not in ["neuroscience", "oncology", "molecular_biology"]:
            claim_texts_combined = " ".join([c.claim_text.lower() for c in rq.generated_claims])
            for bio_term in BIOMEDICAL_TERMS:
                assert bio_term not in claim_texts_combined, (
                    f"Non-biomedical question '{question}' generated claim with unrelated term '{bio_term}'"
                )

        all_generated_claims_by_question[question] = [c.claim_text for c in rq.generated_claims]

    # 5. Verify claims differ between questions
    q_texts = list(all_generated_claims_by_question.keys())
    for i in range(len(q_texts)):
        for j in range(i + 1, len(q_texts)):
            claims1 = set(all_generated_claims_by_question[q_texts[i]])
            claims2 = set(all_generated_claims_by_question[q_texts[j]])
            overlap = claims1.intersection(claims2)
            assert len(overlap) == 0, f"Questions '{q_texts[i]}' and '{q_texts[j]}' produced overlapping claims: {overlap}"


def test_validator_rejects_generic_and_malformed_claims():
    """Verify validator rejects generic boilerplate, malformed phrases, and domain-polluted claims."""
    question = "Does machine learning predict educational outcomes?"

    # Generic boilerplate claim -> MUST REJECT
    c_generic = GeneratedClaim(
        claim_id="c_bad_1",
        question_id="q1",
        claim_text="Empirical evidence demonstrates a direct quantitative relationship involving machine learning."
    )
    is_valid, err = validate_generated_claim(c_generic, research_question=question)
    assert is_valid is False
    assert "generic" in err.lower()

    # Ungrammatical concatenated claim -> MUST REJECT
    c_malformed = GeneratedClaim(
        claim_id="c_bad_2",
        question_id="q1",
        claim_text="Modulation of key pathways associated with machine learning predict educational significantly alters downstream functional responses."
    )
    is_valid, err = validate_generated_claim(c_malformed, research_question=question)
    assert is_valid is False

    # Domain-polluted claim -> MUST REJECT
    c_polluted = GeneratedClaim(
        claim_id="c_bad_3",
        question_id="q1",
        claim_text="Machine learning models alter cell apoptosis and prefrontal cortex responses in students."
    )
    is_valid, err = validate_generated_claim(c_polluted, research_question=question)
    assert is_valid is False
    assert "biomedical" in err.lower()

    # Zero semantic overlap claim -> MUST REJECT
    c_zero_overlap = GeneratedClaim(
        claim_id="c_bad_4",
        question_id="q1",
        claim_text="Supernova explosions emit high energy gamma ray bursts in distant galaxies."
    )
    is_valid, err = validate_generated_claim(c_zero_overlap, research_question=question)
    assert is_valid is False
    assert "zero semantic concept overlap" in err.lower()


def test_semantic_question_parser():
    """Verify question parser correctly extracts entities across different question structures."""
    res1 = parse_scientific_question("What is the relationship between sleep duration and cognitive performance?")
    assert res1["pattern"] == "relationship"
    assert "sleep duration" in res1["var_x"].lower()
    assert "cognitive performance" in res1["var_y"].lower()

    res2 = parse_scientific_question("How does model size affect the accuracy and computational cost of large language models?")
    assert res2["pattern"] == "causal_effect"
    assert "model size" in res2["var_x"].lower()
    assert "accuracy" in res2["var_y1"].lower()
    assert "computational cost" in res2["var_y2"].lower()

    res3 = parse_scientific_question("What factors influence solar cell efficiency?")
    assert res3["pattern"] == "factors_influencing"
    assert "solar cell efficiency" in res3["var_y"].lower()
