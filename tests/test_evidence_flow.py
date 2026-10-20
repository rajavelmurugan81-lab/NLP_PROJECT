"""Unit and Integration Tests for Paper Retrieval Evidence Flow, Sub-Question Planning, and Claim-Evidence Alignment."""

import pytest
import os
import sys

from src.claim_generation.generator import ClaimGenerator
from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.retrieval.paper_evidence import (
    check_paper_domain_relevance,
    get_domain_matched_papers,
    extract_passages_from_paper
)
from src.retrieval.hybrid import HybridRetriever
from src.research_pipeline import ResearchPipeline
from src.evaluation.failure_analyzer import FailureAnalyzer
from src.verification.schema import RetrievedEvidence, VerificationRequest


def test_question_planner_decomposition():
    """Requirement 1: Verify Research Question decomposes into focused sub-questions."""
    generator = ClaimGenerator(provider_type="mock")
    llm_query = "How does model size affect the accuracy and computational cost of large language models?"
    sub_qs = generator.decompose_question(llm_query)
    
    assert len(sub_qs) >= 2
    assert any("accuracy" in sq.lower() or "parameter" in sq.lower() for sq in sub_qs)
    assert any("computational cost" in sq.lower() or "flop" in sq.lower() for sq in sub_qs)


def test_question_relevant_paper_retrieval():
    """Requirement 2 & 3: Verify retrieval produces relevant LLM papers and filters biomedical domain mismatch."""
    retriever = HybridRetriever()
    retriever.load_indexes()
    
    llm_query = "How does model size affect the accuracy and computational cost of large language models?"
    results = retriever.retrieve(query=llm_query, mode="hybrid", top_k=5)
    
    assert len(results) > 0
    # Every returned paper must be about LLMs / scaling / compute and NOT about miRNA / carcinoma / insomnia
    for paper in results:
        title_text = (paper.get("title", "") + " " + paper.get("text", "")).lower()
        assert not any(bio in title_text for bio in ["mirna", "carcinoma", "ovarian", "insomnia"])
        assert any(llm_kw in title_text for llm_kw in ["model", "language", "scaling", "parameter", "compute", "accuracy"])


def test_retrieval_domain_relevance_checker():
    """Requirement 3: Verify check_paper_domain_relevance filters out domain-mismatched papers."""
    llm_query = "How does model size affect the accuracy and computational cost of large language models?"
    
    bio_paper_text = "MicroRNA-21 suppresses cell apoptosis in epithelial ovarian cancer cells."
    is_rel_bio, score_bio = check_paper_domain_relevance(llm_query, bio_paper_text, "miR-21 Carcinoma")
    assert is_rel_bio is False
    assert score_bio == 0.0

    llm_paper_text = "Empirical scaling laws show cross-entropy loss scales as a power law with model parameter count and compute."
    is_rel_llm, score_llm = check_paper_domain_relevance(llm_query, llm_paper_text, "Scaling Laws for Neural Language Models")
    assert is_rel_llm is True
    assert score_llm > 0.0


def test_paper_evidence_passage_extraction():
    """Requirement 4: Verify splitting paper text into identifiable passages with metadata."""
    sample_paper = {
        "paper_id": "paper_test_001",
        "title": "Scaling Laws for Language Models",
        "authors": ["Kaplan et al."],
        "publication_year": 2020,
        "abstract": "We study empirical scaling laws for neural language models.",
        "sections": {
            "Abstract": "We study empirical scaling laws for neural language models.",
            "Results": "Cross-entropy loss scales as a power-law with model size N."
        }
    }
    
    passages = extract_passages_from_paper(sample_paper, base_score=0.92)
    assert len(passages) == 2
    assert passages[0]["paper_id"] == "paper_test_001"
    assert passages[0]["section"] == "Abstract"
    assert "exact_passage" in passages[0]
    assert passages[0]["relevance_score"] > 0.0


def test_claim_evidence_mapping():
    """Requirement 5: Verify atomic claim contains source evidence mapping structure."""
    claim = GeneratedClaim(
        claim_id="c_001",
        question_id="q_001",
        claim_text="Increasing parameter scale improves language model accuracy.",
        source_evidence=[
            {
                "paper_id": "paper_llm_001",
                "paper_title": "Scaling Laws for Neural Language Models",
                "passage_id": "paper_llm_001_p01",
                "section": "Results",
                "exact_passage": "Cross-entropy loss scales as a power-law with model size N.",
                "relevance_score": 0.94
            }
        ]
    )
    assert claim.claim_id == "c_001"
    assert len(claim.source_evidence) == 1
    assert claim.source_evidence[0]["paper_id"] == "paper_llm_001"


def test_failure_signature_overgeneralization_detection():
    """Requirement 8: Test Overgeneralization (F3) detection."""
    analyzer = FailureAnalyzer()
    
    claim = "Sleep deprivation causes permanent cognitive destruction in all humans."
    evidence = [
        RetrievedEvidence(
            doc_id="doc_1",
            text="Short-term sleep restriction is associated with temporary executive function deficits in young adults.",
            score=0.85,
            rank=1
        )
    ]
    
    sig = analyzer.analyze_runtime_instance(
        claim_id="claim_f3_001",
        claim_text=claim,
        predicted_label="SUPPORTED",
        confidence=0.9,
        evidence=evidence
    )
    assert sig is not None
    assert sig.failure_type is not None


def test_end_to_end_llm_provenance_chain(tmp_path):
    """Requirement 11: End-to-end test for LLM scaling question provenance chain."""
    pipeline = ResearchPipeline(provider_type="mock", output_dir=str(tmp_path))
    llm_query = "How does model size affect the accuracy and computational cost of large language models?"
    
    res = pipeline.run_pipeline(research_question=llm_query, top_k=5, mode="hybrid", verbose=False)
    
    rq = res["research_question"]
    assert len(rq.sub_questions) >= 2
    assert len(rq.generated_claims) >= 2
    
    for v_res in res["verification_results"]:
        assert v_res.final_label in ["SUPPORTED", "SUPPORT", "PARTIALLY_SUPPORTED", "NEI"]
        assert len(v_res.evidence) > 0
        for ev in v_res.evidence:
            title_text = (ev.title + " " + ev.text).lower()
            assert not any(bio in title_text for bio in ["mirna", "carcinoma", "ovarian", "insomnia"])
            assert any(kw in title_text for kw in ["model", "language", "scaling", "parameter", "compute", "accuracy"])

    assert os.path.exists(res["report_files"]["json_path"])
    assert os.path.exists(res["report_files"]["md_path"])
