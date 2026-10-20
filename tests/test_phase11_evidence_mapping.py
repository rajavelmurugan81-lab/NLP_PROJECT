"""Unit and integration tests for Phase 11: Passage Extraction, Claim-Evidence Mapping,
Relationship Classification, Evidence Sufficiency, and Provenance Graph Export.
"""

import pytest
import os
import json

from src.retrieval.passage_extractor import PassageExtractor, split_text_into_sentences
from src.verification.evidence_mapper import (
    EvidenceMapper,
    ClaimEvidenceLink,
    classify_relationship
)
from src.claim_generation.generator import ClaimGenerator
from src.claim_generation.schema import GeneratedClaim, ResearchQuestion
from src.verification.schema import (
    VerificationRequest,
    RetrievedEvidence,
    VerificationResult,
    ExecutionMetrics
)
from src.verification.orchestrator import MultiAgentOrchestrator
from src.reporting.generator import ReportGenerator
from src.retrieval.paper_source import UnifiedPaperRetriever


def test_passage_extraction_deterministic_ids():
    """Test passage extraction from paper records with deterministic passage IDs."""
    extractor = PassageExtractor()
    paper = {
        "paper_id": "P001",
        "title": "Sleep Duration and Memory",
        "authors": ["Lo et al."],
        "publication_year": 2020,
        "abstract": "Sleep is important for optimal cognitive functioning across the lifespan. Short sleep duration impairs sustained attention performance. Long sleep duration shows mixed cognitive outcomes.",
        "source": "openalex",
        "doi": "https://doi.org/10.1016/j.sleep.2020.01",
        "relevance_score": 0.95
    }

    passages = extractor.extract_passages_from_paper(paper, paper_index=1)
    assert len(passages) == 3
    assert passages[0]["passage_id"] == "P001_P01"
    assert passages[1]["passage_id"] == "P001_P02"
    assert passages[2]["passage_id"] == "P001_P03"
    assert passages[0]["text"] == "Sleep is important for optimal cognitive functioning across the lifespan."
    assert passages[1]["text"] == "Short sleep duration impairs sustained attention performance."


def test_verbatim_evidence_text_preservation():
    """Test that extracted evidence passages preserve verbatim original wording without paraphrasing."""
    raw_abstract = "CRISPR-Cas9 mediated knock-out of target oncogenes reduces cell proliferation in human cell lines. Off-target cleavage rates are decreased by high-fidelity Cas9 variants."
    sentences = split_text_into_sentences(raw_abstract)
    assert len(sentences) == 2
    assert sentences[0] == "CRISPR-Cas9 mediated knock-out of target oncogenes reduces cell proliferation in human cell lines."
    assert sentences[1] == "Off-target cleavage rates are decreased by high-fidelity Cas9 variants."


def test_claim_evidence_relationship_classification():
    """Test classifying evidence relationships (SUPPORTS, CONTRADICTS, PARTIALLY_SUPPORTS, IRRELEVANT)."""
    claim = "Short sleep duration reduces cognitive performance in adults."

    # Supporting passage
    p_sup = "Short sleep duration reduces cognitive performance and sustained attention in adults."
    rel_sup, score_sup = classify_relationship(claim, p_sup)
    assert rel_sup in ["SUPPORTS", "PARTIALLY_SUPPORTS"]
    assert score_sup >= 0.60

    # Contradicting passage
    p_ref = "Short sleep duration does not reduce cognitive performance in healthy young adults."
    rel_ref, score_ref = classify_relationship(claim, p_ref)
    assert rel_ref == "CONTRADICTS"

    # Irrelevant passage
    p_irr = "MicroRNA-21 suppresses cell apoptosis in ovarian carcinoma cell lines."
    rel_irr, score_irr = classify_relationship(claim, p_irr)
    assert rel_irr == "IRRELEVANT"


def test_evidence_sufficiency_evaluation():
    """Test evidence sufficiency criteria evaluation and fallback to INSUFFICIENT_EVIDENCE."""
    mapper = EvidenceMapper()

    # Sufficient supporting evidence link
    link_sup = ClaimEvidenceLink(
        claim_id="c1",
        paper_id="p1",
        passage_id="P001_P01",
        relationship="SUPPORTS",
        evidence_text="Short sleep duration impairs executive function.",
        paper_title="Sleep Study",
        doi="10.1000/1",
        source="openalex",
        retrieval_score=0.90,
        evidence_score=0.85,
        confidence=0.87
    )

    is_suff, metrics = mapper.evaluate_sufficiency([link_sup], min_passages=1, min_score=0.50)
    assert is_suff is True
    assert metrics["supporting_count"] == 1

    # Insufficient evidence links (empty list)
    is_suff_empty, metrics_empty = mapper.evaluate_sufficiency([], min_passages=1, min_score=0.50)
    assert is_suff_empty is False
    assert metrics_empty["total_passages"] == 0


def test_claim_evidence_mapping_graph():
    """Test mapping an atomic claim against candidate passages into structured links."""
    mapper = EvidenceMapper()
    claim_id = "c001"
    claim_text = "Increasing parameter scale in large language models improves task accuracy."

    passages = [
        {
            "paper_id": "paper_llm_1",
            "passage_id": "P001_P01",
            "text": "Cross-entropy loss scales as a power-law with model size N and dataset size D, improving accuracy.",
            "title": "Scaling Laws for Neural Language Models",
            "doi": "10.48550/arxiv.2001.08361",
            "source": "openalex",
            "relevance_score": 0.95
        },
        {
            "paper_id": "paper_bio_1",
            "passage_id": "P002_P01",
            "text": "MicroRNA-21 suppresses apoptosis in epithelial ovarian cancer cells.",
            "title": "miRNA in Cancer",
            "doi": "10.1000/bio",
            "source": "local_scifact",
            "relevance_score": 0.20
        }
    ]

    links = mapper.map_claim_to_passages(claim_id, claim_text, passages, top_k=5)
    assert len(links) == 1
    assert links[0].paper_id == "paper_llm_1"
    assert links[0].passage_id == "P001_P01"
    assert links[0].relationship in ["SUPPORTS", "PARTIALLY_SUPPORTS"]


def test_multi_agent_verification_provenance_integration():
    """Test multi-agent orchestrator executing verification with exact passage provenance."""
    orchestrator = MultiAgentOrchestrator()
    req = VerificationRequest(
        claim_id="c_test_1",
        claim_text="Adults regularly sleeping fewer than seven hours demonstrate reduced cognitive performance.",
        retrieval_mode="hybrid",
        top_k=3
    )

    # Pre-built evidence with exact passage metadata
    evidence = [
        RetrievedEvidence(
            doc_id="doc_1",
            text="Adults sleeping under seven hours per night show significantly impaired executive performance.",
            score=0.92,
            rank=1,
            title="Sleep and Performance",
            source="openalex",
            metadata={
                "paper_id": "P001",
                "passage_id": "P001_P01",
                "exact_passage": "Adults sleeping under seven hours per night show significantly impaired executive performance.",
                "authors": "Lo et al.",
                "publication_year": 2020,
                "url_doi": "https://doi.org/10.1016/sleep"
            }
        )
    ]

    res = orchestrator.verify(request=req, pre_retrieved_evidence=evidence)
    assert res.claim_id == "c_test_1"
    assert res.final_label in ["SUPPORT", "SUPPORTED", "NEI"]
    assert len(res.evidence) == 1
    assert res.evidence[0].metadata["passage_id"] == "P001_P01"


def test_report_generator_provenance_graph_export(tmp_path):
    """Test ReportGenerator exporting full provenance graph in JSON and Markdown outputs."""
    report_gen = ReportGenerator(output_dir=str(tmp_path))
    
    rq = ResearchQuestion(
        question_id="q_phase11",
        research_question="What is the relationship between sleep duration and cognitive performance?",
        sub_questions=["What is the effect of sleep duration on executive function?"],
        generated_claims=[
            GeneratedClaim(
                claim_id="c_p11_01",
                question_id="q_phase11",
                claim_text="Short sleep duration reduces cognitive performance.",
                status="VERIFIED_SUPPORTED"
            )
        ]
    )

    res = VerificationResult(
        claim_id="c_p11_01",
        claim_text="Short sleep duration reduces cognitive performance.",
        final_label="SUPPORTED",
        confidence=0.88,
        evidence=[
            RetrievedEvidence(
                doc_id="doc_1",
                text="Short sleep duration impairs cognitive performance in older adults.",
                score=0.92,
                rank=1,
                title="Sleep and Cognition Review",
                source="openalex",
                metadata={
                    "paper_id": "P001",
                    "passage_id": "P001_P02",
                    "exact_passage": "Short sleep duration impairs cognitive performance in older adults.",
                    "authors": "Lo et al.",
                    "publication_year": 2015,
                    "url_doi": "https://doi.org/10.1016/sleep.2015"
                }
            )
        ]
    )

    out_files = report_gen.generate_report(question=rq, verification_results=[res])
    assert os.path.exists(out_files["json_path"])
    assert os.path.exists(out_files["md_path"])

    with open(out_files["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["question_id"] == "q_phase11"
    assert len(data["claims"]) == 1
    ev_item = data["claims"][0]["evidence"][0]
    assert ev_item["paper_id"] == "P001"
    assert ev_item["passage_id"] == "P001_P02"
    assert ev_item["exact_passage"] == "Short sleep duration impairs cognitive performance in older adults."
