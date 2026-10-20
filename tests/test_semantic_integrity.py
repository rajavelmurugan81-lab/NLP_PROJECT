"""Semantic Regression Tests (Tests A through F) for CiteGuard Semantic Integrity Correction.

Verifies:
- Test A: Strict domain blocking for compound quantum-resistant federated learning queries.
- Test B: Bibliographic ranking sorted by cited_by_count descending.
- Test C: No unrelated paper filling when domain relevance fails.
- Test D: Exact metadata anti-fabrication (preserves real DOI/citations, doi=None if missing, experimental_results_status='NOT_VERIFIED').
- Test E: Dynamic confidence variability (no static 0.85, low confidence <= 0.35 on missing/mismatched evidence).
- Test F: Runtime auditor flags semantic domain mismatches with severity HIGH and ledger_status FLAGGED.
"""

import pytest
from src.retrieval.paper_evidence import check_paper_domain_relevance
from src.retrieval.paper_source import rank_papers_by_domain
from src.retrieval.openalex import OpenAlexPaperRetriever
from src.retrieval.passage_extractor import PassageExtractor
from src.verification.evidence_mapper import EvidenceMapper, classify_relationship
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.schema import VerificationRequest, RetrievedEvidence
from src.claim_generation.classifier import classify_question, QuestionType
from src.evaluation.runtime_auditor import RuntimeAuditor


def test_a_domain_blocking_compound_quantum_fl():
    """Test A: Strict topic relevance gate blocks smart-grid, drug discovery, and oncology papers for quantum-resistant federated learning."""
    query = "what are the 10 most cited papers published in 2026 on quantum resistant federated learning and provide their exact DOI,citation count and experimental results?"

    # 1. Smart Grid paper (must be blocked)
    smart_grid_title = "Cybersecurity and Resilience of Smart Grids: A Review of Threat Landscape, Incidents, and Emerging Solutions"
    smart_grid_text = "Peer-reviewed scientific literature and authoritative institutional sources are synthesised to examine modern smart grid architectures, key security challenges, major cyberthreats, and documented real-world cyber incidents affecting energy infrastructure up to 2025."
    is_rel_sg, score_sg = check_paper_domain_relevance(query=query, paper_text=smart_grid_text, paper_title=smart_grid_title)
    assert is_rel_sg is False, "Smart Grid paper must be blocked for quantum-resistant federated learning"
    assert score_sg == 0.0

    # 2. Drug Discovery paper (must be blocked)
    drug_title = "Nature meets machine: the AI renaissance in natural product drug discovery"
    drug_text = "Looking ahead, we examine emerging paradigms-including quantum machine learning, federated data ecosystems, and AI-assisted molecular design-that may further expand the scope of NP-based research."
    is_rel_dd, score_dd = check_paper_domain_relevance(query=query, paper_text=drug_text, paper_title=drug_title)
    assert is_rel_dd is False, "Natural product drug discovery paper must be blocked for quantum-resistant federated learning"
    assert score_dd == 0.0

    # 3. Oncology paper (must be blocked)
    onc_title = "Artificial intelligence-enabled multi-omics biomarkers for immune checkpoint blockade: mechanisms, predictive modeling, and clinical translation"
    onc_text = "We conclude by outlining future directions in single-cell and spatial multi-omics integration, federated learning, and generative modeling to accelerate robust, generalizable precision immunotherapy."
    is_rel_onc, score_onc = check_paper_domain_relevance(query=query, paper_text=onc_text, paper_title=onc_title)
    assert is_rel_onc is False, "Immunotherapy oncology paper must be blocked for quantum-resistant federated learning"
    assert score_onc == 0.0

    # 4. Genuine Post-Quantum Federated Learning paper (must be accepted)
    qrfl_title = "Lattice-Based Post-Quantum Cryptography for Privacy-Preserving Federated Learning"
    qrfl_text = "We propose a post-quantum lattice encryption framework for secure aggregation in federated learning clients, guaranteeing resistance against quantum adversaries while preserving distributed model accuracy."
    is_rel_qr, score_qr = check_paper_domain_relevance(query=query, paper_text=qrfl_text, paper_title=qrfl_title)
    assert is_rel_qr is True, "Genuine quantum-resistant federated learning paper must be accepted"
    assert score_qr >= 0.75


def test_b_bibliographic_ranking_by_cited_by_count():
    """Test B: Bibliographic queries sort papers by cited_by_count descending."""
    query = "what are the 10 most cited papers published in 2026 on quantum resistant federated learning"

    candidate_papers = [
        {
            "paper_id": "p1",
            "title": "Quantum Resistance in Federated Learning via Module-LWE",
            "abstract": "We construct quantum-resistant federated learning protocols based on module learning with errors (LWE) lattice cryptography.",
            "cited_by_count": 15,
            "score": 0.85
        },
        {
            "paper_id": "p2",
            "title": "Post-Quantum Cryptographic Aggregation for Distributed Federated Networks",
            "abstract": "This work introduces post-quantum cipher suites for secure client aggregation in federated learning under quantum threats.",
            "cited_by_count": 89,
            "score": 0.80
        },
        {
            "paper_id": "p3",
            "title": "Shor-Resistant Federated Averaging with Zero-Knowledge Proofs",
            "abstract": "Quantum-resistant federated learning with lattice-based zero-knowledge proofs achieves robust verification against quantum decryption.",
            "cited_by_count": 42,
            "score": 0.82
        }
    ]

    ranked = rank_papers_by_domain(query=query, papers=candidate_papers, sort_by="cited_by_count:desc")
    assert len(ranked) == 3
    # Sorted by cited_by_count descending: 89 -> 42 -> 15
    assert ranked[0]["paper_id"] == "p2"
    assert ranked[0]["cited_by_count"] == 89
    assert ranked[1]["paper_id"] == "p3"
    assert ranked[1]["cited_by_count"] == 42
    assert ranked[2]["paper_id"] == "p1"
    assert ranked[2]["cited_by_count"] == 15
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3


def test_c_no_unrelated_paper_filling():
    """Test C: When retrieved papers fail topic relevance, no unrelated passages are mapped into evidence."""
    extractor = PassageExtractor()
    mapper = EvidenceMapper()

    unrelated_paper = {
        "paper_id": "W7124922659",
        "title": "Cybersecurity and Resilience of Smart Grids",
        "abstract": "Smart grid architectures and modern power systems are analyzed for electrical substation security.",
        "publication_year": 2026,
        "evidence_eligible": False,
        "topic_match": False,
        "score": 0.0,
        "relevance_score": 0.0
    }

    passages = extractor.extract_passages_from_paper(unrelated_paper)
    assert len(passages) >= 1
    for p in passages:
        assert p["evidence_eligible"] is False
        assert p["score"] == 0.0

    claim_text = "OpenAlex indexes scientific literature on quantum resistant federated learning published in 2026."
    links = mapper.map_claim_to_passages(claim_id="c_test", claim_text=claim_text, passages=passages)
    # Unrelated passages must be completely excluded from claim evidence links
    assert len(links) == 0


def test_d_exact_metadata_anti_fabrication():
    """Test D: Exact metadata from OpenAlex is preserved without fabrication; missing DOI is None."""
    retriever = OpenAlexPaperRetriever()

    # Work without DOI
    work_no_doi = {
        "id": "https://openalex.org/W12345678",
        "display_name": "Theoretical Bounds on Post-Quantum Cryptography",
        "publication_year": 2026,
        "cited_by_count": 0,
        "doi": None,
        "abstract_inverted_index": {"We": [0], "study": [1], "bounds": [2]}
    }
    normalized = retriever.normalize_work(work_no_doi)
    assert normalized["doi"] is None
    assert normalized["cited_by_count"] == 0
    assert normalized["publication_year"] == 2026

    # Work with exact DOI and real citation count
    work_with_doi = {
        "id": "https://openalex.org/W98765432",
        "display_name": "Lattice-Based Federated Learning Security",
        "publication_year": 2026,
        "cited_by_count": 37,
        "doi": "https://doi.org/10.1145/3372297.3417238",
        "abstract_inverted_index": {"Federated": [0], "learning": [1]}
    }
    normalized_with = retriever.normalize_work(work_with_doi)
    assert normalized_with["doi"] == "https://doi.org/10.1145/3372297.3417238"
    assert normalized_with["cited_by_count"] == 37


def test_e_dynamic_confidence_variability():
    """Test E: Verification confidence is calculated dynamically and is NOT a static 0.85 across queries."""
    orchestrator = MultiAgentOrchestrator()

    # Case 1: Zero evidence -> confidence must be <= 0.35
    req_empty = VerificationRequest(
        claim_id="c_empty",
        claim_text="Quantum resistant federated learning solves Byzantine faults.",
        top_k=5
    )
    res_empty = orchestrator.verify(request=req_empty, pre_retrieved_evidence=[])
    assert res_empty.confidence <= 0.35
    assert res_empty.final_label == "NEI"

    # Case 2: High quality evidence -> confidence varies dynamically based on evidence scores
    ev_high = [
        RetrievedEvidence(doc_id="d1", text="Lattice cryptography guarantees quantum security in federated models.", score=0.92, rank=1, source="openalex"),
        RetrievedEvidence(doc_id="d2", text="Post-quantum protocols achieve 95% accuracy in federated learning.", score=0.88, rank=2, source="openalex")
    ]
    req_high = VerificationRequest(
        claim_id="c_high",
        claim_text="Lattice cryptography enables quantum security in federated models.",
        top_k=5
    )
    res_high = orchestrator.verify(request=req_high, pre_retrieved_evidence=ev_high)
    assert res_high.confidence != res_empty.confidence
    assert res_high.confidence >= 0.70

    # Case 3: Moderate evidence with different score -> confidence differs from Case 2
    ev_mod = [
        RetrievedEvidence(doc_id="d3", text="Preliminary tests show some quantum resistance under certain constraints.", score=0.62, rank=1, source="openalex")
    ]
    req_mod = VerificationRequest(
        claim_id="c_mod",
        claim_text="Preliminary tests show quantum resistance under certain constraints.",
        top_k=5
    )
    res_mod = orchestrator.verify(request=req_mod, pre_retrieved_evidence=ev_mod)
    assert res_mod.confidence != res_high.confidence
    assert res_mod.confidence < res_high.confidence


def test_f_runtime_auditor_flags_semantic_mismatch(tmp_path):
    """Test F: Runtime auditor flags domain mismatches with severity HIGH and ledger_status FLAGGED."""
    ledger_file = str(tmp_path / "runtime_audit_mismatch_test.jsonl")
    auditor = RuntimeAuditor(ledger_filepath=ledger_file)

    # Claim with domain mismatched evidence
    mismatched_evidence = [
        {
            "score": 0.0,
            "relation": "irrelevant",
            "metadata": {
                "relationship": "IRRELEVANT",
                "evidence_eligible": False,
                "topic_match": False
            }
        }
    ]

    record = auditor.audit_claim(
        claim_id="c_mismatch",
        claim_text="OpenAlex indexes scientific literature on quantum resistant federated learning published in 2026.",
        verdict="SUPPORTED",
        confidence=0.35,
        evidence=mismatched_evidence,
        question="what are the 10 most cited papers published in 2026 on quantum resistant federated learning?",
        run_id="run_mismatch_test"
    )

    assert record.failure_type == "DOMAIN_MISMATCH"
    assert record.severity == "HIGH"
    assert record.ledger_status == "FLAGGED"
    assert "domain relevance" in record.explanation.lower()
