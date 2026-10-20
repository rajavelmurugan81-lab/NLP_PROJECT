"""Comprehensive test suite for Retrieval-First Claim Extraction & GroundedClaim Provenance.

Validates the core research invariant:
NO GROUNDED SCIENTIFIC CLAIM IS CREATED BEFORE RETRIEVAL.

Ensures deterministic provenance, exact evidence text preservation,
claim type categorization, and seamless compatibility across CiteGuard.
"""

import pytest
from src.evaluation.claim_extractor import GroundedClaim, RetrievalFirstClaimExtractor
from src.retrieval.passage_extractor import PassageExtractor
from src.planner.planner_agent import PlannerResult


def test_grounded_claim_dataclass_fields():
    """1. Test GroundedClaim dataclass has all required fields and default values."""
    claim = GroundedClaim(
        claim_id="c_001_p01",
        question_id="q_101",
        claim_text="Short sleep duration impairs sustained attention.",
        source_paper_id="P001",
        source_title="Sleep and Attention",
        passage_id="P001_P01",
        exact_evidence_text="Short sleep duration impairs sustained attention performance.",
        source_doi="https://doi.org/10.1016/sleep.2020",
        evidence_score=0.92,
        retrieval_rank=1,
        claim_type="causal_hypothesis",
        source_year=2020,
        domain="cognitive_neuroscience",
        extraction_confidence=0.95
    )
    assert claim.claim_id == "c_001_p01"
    assert claim.question_id == "q_101"
    assert claim.source_paper_id == "P001"
    assert claim.passage_id == "P001_P01"
    assert claim.source_doi == "https://doi.org/10.1016/sleep.2020"
    assert claim.source_year == 2020
    assert claim.is_valid is True
    assert claim.status == "EXTRACTED"
    assert claim.extraction_confidence == 0.95


def test_grounded_claim_serialization():
    """2. Test GroundedClaim to_dict and from_dict roundtrip."""
    claim = GroundedClaim(
        claim_id="c_002_p02",
        question_id="q_102",
        claim_text="CRISPR knock-out reduces cancer cell proliferation.",
        source_paper_id="P002",
        source_title="CRISPR Oncology",
        passage_id="P002_P01",
        exact_evidence_text="CRISPR-Cas9 knockout reduces cancer cell proliferation by 45%.",
        source_doi="10.1000/crispr",
        evidence_score=0.88,
        retrieval_rank=1,
        claim_type="statistical_finding",
        source_year=2023,
        domain="oncology"
    )
    d = claim.to_dict()
    assert isinstance(d, dict)
    assert d["claim_id"] == "c_002_p02"
    assert d["claim_type"] == "statistical_finding"
    assert d["provenance"]["paper_id"] == "P002"

    rebuilt = GroundedClaim.from_dict(d)
    assert rebuilt.claim_id == claim.claim_id
    assert rebuilt.claim_text == claim.claim_text
    assert rebuilt.source_doi == claim.source_doi
    assert rebuilt.evidence_score == claim.evidence_score


def test_grounded_claim_compatibility_with_generated_claim():
    """3. Test that GroundedClaim satisfies the interface expected by GeneratedClaim consumers."""
    claim = GroundedClaim(
        claim_id="c_compat",
        question_id="q_compat",
        claim_text="Transformer parameter scaling obeys power laws.",
        source_paper_id="P003",
        source_title="Scaling Laws",
        passage_id="P003_P01",
        exact_evidence_text="Transformer parameter scaling obeys empirical power laws.",
        evidence_score=0.91,
        retrieval_rank=1
    )
    assert hasattr(claim, "claim_id")
    assert hasattr(claim, "claim_text")
    assert hasattr(claim, "claim_type")
    assert hasattr(claim, "is_valid")
    assert hasattr(claim, "status")
    assert hasattr(claim, "source_evidence")
    assert hasattr(claim, "metadata")
    assert len(claim.source_evidence) > 0
    assert claim.source_evidence[0]["passage_id"] == "P003_P01"


def test_provenance_structure_complete():
    """4. Test that the provenance dictionary contains all required fields."""
    claim = GroundedClaim(
        claim_id="c_prov",
        question_id="q_prov",
        claim_text="Metformin reduces cardiovascular events in diabetics.",
        source_paper_id="W1234567890",
        source_title="Metformin Trial",
        passage_id="P001_P04",
        exact_evidence_text="Metformin reduces cardiovascular events by 18% in type 2 diabetics.",
        source_doi="10.1056/NEJMoa012345",
        evidence_score=0.94,
        retrieval_rank=1,
        source_year=2021
    )
    prov = claim.provenance
    assert "source" in prov
    assert "paper_id" in prov
    assert "passage_id" in prov
    assert "doi" in prov
    assert "title" in prov
    assert "publication_year" in prov
    assert "exact_passage" in prov
    assert "evidence_score" in prov
    assert "retrieval_rank" in prov
    assert "provenance_status" in prov
    assert prov["provenance_status"] == "VERIFIED"
    assert prov["passage_id"] == "P001_P04"
    assert prov["exact_passage"] == claim.exact_evidence_text


def test_extractor_initialization():
    """5. Test initializing RetrievalFirstClaimExtractor."""
    extractor = RetrievalFirstClaimExtractor(min_confidence=0.45, default_domain="biomedicine")
    assert extractor.min_confidence == 0.45
    assert extractor.default_domain == "biomedicine"
    assert isinstance(extractor.passage_extractor, PassageExtractor)


def test_extract_claims_from_passages():
    """6. Test extracting grounded claims from pre-extracted candidate passages."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [
        {
            "paper_id": "P001",
            "passage_id": "P001_P01",
            "text": "Short sleep duration impairs sustained attention performance in working adults.",
            "title": "Sleep and Performance",
            "doi": "10.1016/sleep",
            "relevance_score": 0.95,
            "publication_year": 2020,
            "evidence_eligible": True,
            "topic_match": True
        },
        {
            "paper_id": "P001",
            "passage_id": "P001_P02",
            "text": "Chronic sleep restriction alters prefrontal cortex functional connectivity.",
            "title": "Sleep and Performance",
            "doi": "10.1016/sleep",
            "relevance_score": 0.90,
            "publication_year": 2020,
            "evidence_eligible": True,
            "topic_match": True
        }
    ]

    claims = extractor.extract_claims(
        research_question="What is the relationship between sleep duration and cognitive performance?",
        passages=passages,
        top_k=5
    )
    assert len(claims) == 2
    assert all(isinstance(c, GroundedClaim) for c in claims)
    assert claims[0].source_paper_id == "P001"
    passage_ids = [c.passage_id for c in claims]
    assert "P001_P01" in passage_ids
    assert "P001_P02" in passage_ids


def test_no_claims_created_before_retrieval():
    """7. Critical Invariant: No grounded claim is created before retrieval."""
    extractor = RetrievalFirstClaimExtractor()
    claims = extractor.extract_claims(
        research_question="What is the impact of exercise on hippocampal neurogenesis?",
        papers=[],
        passages=[]
    )
    assert claims == []
    assert len(claims) == 0


def test_exact_evidence_text_unmodified():
    """8. Research Integrity: Exact evidence text is preserved verbatim without paraphrasing."""
    extractor = RetrievalFirstClaimExtractor()
    verbatim_text = "Analysis revealed that miRNA-155 is upregulated 3.4-fold (p < 0.01) in activated macrophages."
    passages = [{
        "paper_id": "P005",
        "passage_id": "P005_P01",
        "text": verbatim_text,
        "title": "miRNA Regulation",
        "relevance_score": 0.92,
        "evidence_eligible": True,
        "topic_match": True
    }]
    claims = extractor.extract_claims(
        research_question="Does miRNA-155 regulate macrophage activation?",
        passages=passages
    )
    assert len(claims) == 1
    assert claims[0].exact_evidence_text == verbatim_text
    assert claims[0].provenance["exact_passage"] == verbatim_text


def test_deterministic_claim_ids():
    """9. Test deterministic claim IDs linked to passage identifiers."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [{
        "paper_id": "P001",
        "passage_id": "P001_P03",
        "text": "Sleep spindle density predicts memory consolidation accuracy in young adults.",
        "title": "Sleep and Memory",
        "relevance_score": 0.89,
        "evidence_eligible": True,
        "topic_match": True
    }]
    claims = extractor.extract_claims(
        research_question="How does sleep affect memory?",
        passages=passages
    )
    assert len(claims) == 1
    assert "p001_p03" in claims[0].claim_id.lower()


def test_passage_id_linkage():
    """10. Test passage ID linkage integrity."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [
        {"paper_id": "P100", "passage_id": "P100_P01", "text": "Model quantization reduces GPU memory footprint.", "relevance_score": 0.85, "title": "Quantization", "evidence_eligible": True, "topic_match": True},
        {"paper_id": "P100", "passage_id": "P100_P02", "text": "Post-training 4-bit quantization preserves 99% of perplexity.", "relevance_score": 0.87, "title": "Quantization", "evidence_eligible": True, "topic_match": True}
    ]
    claims = extractor.extract_claims("Quantization efficiency", passages=passages)
    assert len(claims) == 2
    assigned_passage_ids = {c.passage_id for c in claims}
    assert assigned_passage_ids == {"P100_P01", "P100_P02"}


def test_multi_paper_provenance():
    """11. Test extraction across multiple distinct papers accurately retains distinct paper IDs."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [
        {"paper_id": "P001", "passage_id": "P001_P01", "text": "Aerobic exercise increases serum BDNF levels in humans.", "relevance_score": 0.90, "title": "BDNF Study 1", "evidence_eligible": True, "topic_match": True},
        {"paper_id": "P002", "passage_id": "P002_P01", "text": "High-intensity interval training enhances neuroplasticity biomarkers.", "relevance_score": 0.88, "title": "HIIT Study 2", "evidence_eligible": True, "topic_match": True}
    ]
    claims = extractor.extract_claims("Exercise and neuroplasticity", passages=passages)
    assert len(claims) == 2
    paper_ids = [c.source_paper_id for c in claims]
    assert "P001" in paper_ids
    assert "P002" in paper_ids


def test_top_k_parameter_honored():
    """12. Test that top_k restricts the maximum number of extracted claims."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [
        {"paper_id": f"P00{i}", "passage_id": f"P00{i}_P01", "text": f"Distinct scientific assertion number {i} regarding cognitive mechanisms.", "relevance_score": 0.90 - i * 0.05, "title": f"Paper {i}", "evidence_eligible": True, "topic_match": True}
        for i in range(1, 6)
    ]
    claims_2 = extractor.extract_claims("Cognitive mechanisms", passages=passages, top_k=2)
    assert len(claims_2) == 2
    claims_4 = extractor.extract_claims("Cognitive mechanisms", passages=passages, top_k=4)
    assert len(claims_4) == 4


def test_claim_type_categorization():
    """13. Test categorization into statistical_finding, causal_hypothesis, and empirical_assertion."""
    extractor = RetrievalFirstClaimExtractor()

    # Statistical
    t_stat = extractor.classify_claim_type("Patients treated with drug X showed 35% reduction in mortality (p < 0.01).")
    assert t_stat == "statistical_finding"

    # Causal
    t_causal = extractor.classify_claim_type("Sleep deprivation impairs working memory function.")
    assert t_causal == "causal_hypothesis"

    # Empirical assertion
    t_emp = extractor.classify_claim_type("The hippocampus is located in the medial temporal lobe.")
    assert t_emp == "empirical_assertion"


def test_filtering_ineligible_passages():
    """14. Test filtering of passages flagged as ineligible or non-topic matches."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [
        {"paper_id": "P001", "passage_id": "P001_P01", "text": "Sleep duration influences cognition.", "relevance_score": 0.90, "evidence_eligible": True, "topic_match": True, "title": "Paper 1"},
        {"paper_id": "P002", "passage_id": "P002_P01", "text": "MicroRNA regulates cell apoptosis in ovaries.", "relevance_score": 0.10, "evidence_eligible": False, "topic_match": False, "title": "Paper 2"},
        {"paper_id": "P003", "passage_id": "P003_P01", "text": "Another irrelevant sentence with zero score.", "relevance_score": 0.0, "evidence_eligible": True, "topic_match": True, "title": "Paper 3"}
    ]
    claims = extractor.extract_claims("Sleep duration and cognition", passages=passages)
    assert len(claims) == 1
    assert claims[0].source_paper_id == "P001"


def test_research_plan_subquestion_alignment():
    """15. Test that PlannerResult research subquestions guide passage scoring."""
    extractor = RetrievalFirstClaimExtractor()
    plan = PlannerResult(
        original_question="How does sleep affect cognitive performance?",
        research_objective="Evaluate sleep impacts on executive function",
        question_type="SCIENTIFIC_HYPOTHESIS",
        domain_constraints=["neuroscience"],
        research_subquestions=[
            "How does sleep deprivation affect working memory?",
            "What neural circuits mediate attention deficits?"
        ],
        retrieval_queries=["sleep deprivation working memory prefrontal"]
    )
    passages = [
        {"paper_id": "P001", "passage_id": "P001_P01", "text": "Sleep deprivation impairs working memory through prefrontal dysfunction.", "relevance_score": 0.85, "title": "Sleep 1", "evidence_eligible": True, "topic_match": True},
        {"paper_id": "P002", "passage_id": "P002_P01", "text": "General biology of circadian sleep architecture in animals.", "relevance_score": 0.85, "title": "Sleep 2", "evidence_eligible": True, "topic_match": True}
    ]
    claims = extractor.extract_claims(
        research_question="How does sleep affect cognitive performance?",
        research_plan=plan,
        passages=passages,
        top_k=1
    )
    assert len(claims) == 1
    assert claims[0].source_paper_id == "P001"


def test_contradictory_passages_preserved():
    """16. Test that contradictory passages are preserved as candidate claims rather than silently dropped."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [
        {"paper_id": "P001", "passage_id": "P001_P01", "text": "Caffeine improves vigilance during sustained attention tasks.", "relevance_score": 0.90, "title": "Caffeine 1", "evidence_eligible": True, "topic_match": True},
        {"paper_id": "P002", "passage_id": "P002_P01", "text": "Caffeine does not improve vigilance in chronically sleep-deprived individuals.", "relevance_score": 0.88, "title": "Caffeine 2", "evidence_eligible": True, "topic_match": True}
    ]
    claims = extractor.extract_claims("Does caffeine improve vigilance during sleep loss?", passages=passages, top_k=2)
    assert len(claims) == 2
    claim_texts = [c.claim_text for c in claims]
    assert any("improves" in ct for ct in claim_texts)
    assert any("does not improve" in ct for ct in claim_texts)


def test_source_doi_and_year_preservation():
    """17. Test that source DOI and publication year are preserved in the claim provenance."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [{
        "paper_id": "P888",
        "passage_id": "P888_P01",
        "text": "Deep residual learning accelerates neural network optimization.",
        "title": "Deep Residual Learning",
        "doi": "10.1109/CVPR.2016.90",
        "publication_year": 2016,
        "relevance_score": 0.96,
        "evidence_eligible": True,
        "topic_match": True
    }]
    claims = extractor.extract_claims("Residual networks convergence", passages=passages)
    assert len(claims) == 1
    assert claims[0].source_doi == "10.1109/CVPR.2016.90"
    assert claims[0].source_year == 2016
    assert claims[0].provenance["doi"] == "10.1109/CVPR.2016.90"
    assert claims[0].provenance["publication_year"] == 2016


def test_clean_claim_statement_synthesis():
    """18. Test stripping introductory conversational meta-discourse while keeping claim intact."""
    extractor = RetrievalFirstClaimExtractor()
    framed_text = "In this study, we found that regular aerobic exercise increases neurogenesis in the dentate gyrus."
    cleaned = extractor.clean_claim_statement(framed_text)
    assert cleaned == "Regular aerobic exercise increases neurogenesis in the dentate gyrus."

    framed_text_2 = "Our results indicate that model scaling follows an inverse power-law loss trajectory."
    cleaned_2 = extractor.clean_claim_statement(framed_text_2)
    assert cleaned_2 == "Model scaling follows an inverse power-law loss trajectory."


def test_extraction_confidence_bounds():
    """19. Test that extraction_confidence remains strictly within [0.40, 1.0]."""
    extractor = RetrievalFirstClaimExtractor()
    passages = [{
        "paper_id": "P001",
        "passage_id": "P001_P01",
        "text": "Sleep duration influences working memory capacity in healthy adults.",
        "relevance_score": 0.95,
        "title": "Sleep",
        "evidence_eligible": True,
        "topic_match": True
    }]
    claims = extractor.extract_claims("Sleep duration and working memory", passages=passages)
    assert len(claims) == 1
    assert 0.40 <= claims[0].extraction_confidence <= 1.0


def test_integration_with_passage_extractor():
    """20. Test end-to-end integration: Raw paper -> PassageExtractor -> RetrievalFirstClaimExtractor."""
    raw_papers = [
        {
            "paper_id": "P001",
            "title": "Sleep and Cognitive Performance",
            "authors": ["Lo et al."],
            "publication_year": 2020,
            "abstract": "Sleep is crucial for cognitive performance. Short sleep duration impairs sustained attention performance. Long sleep duration demonstrates inconsistent results.",
            "source": "openalex",
            "doi": "https://doi.org/10.1016/sleep.2020.01",
            "relevance_score": 0.95
        }
    ]

    passage_ext = PassageExtractor()
    passages = passage_ext.extract_passages_from_papers(raw_papers)
    assert len(passages) == 3

    claim_ext = RetrievalFirstClaimExtractor()
    grounded_claims = claim_ext.extract_claims(
        research_question="What is the relationship between sleep duration and cognitive performance?",
        passages=passages,
        top_k=2
    )

    assert len(grounded_claims) == 2
    for gc in grounded_claims:
        assert gc.source_paper_id == "P001"
        assert gc.source_doi == "https://doi.org/10.1016/sleep.2020.01"
        assert gc.source_year == 2020
        assert gc.passage_id in ["P001_P01", "P001_P02", "P001_P03"]
        assert gc.exact_evidence_text in [p["text"] for p in passages]
        assert gc.provenance["provenance_status"] == "VERIFIED"
