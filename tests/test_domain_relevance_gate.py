"""Tests for Domain Relevance Gate and Thresholds in CiteGuard."""

import pytest
from src.retrieval.paper_evidence import check_paper_domain_relevance
from src.retrieval.paper_source import rank_papers_by_domain
from src.verification.evidence_mapper import (
    classify_relationship,
    HIGH_RELEVANCE_THRESHOLD,
    SUPPORT_THRESHOLD,
    PARTIAL_THRESHOLD
)


def test_domain_relevance_gate_blocks_bio_for_quantum():
    query = "quantum resistant federated learning"
    bio_text = "MicroRNA-21 regulates apoptosis and promotes tumor cell invasion in breast carcinoma patients."
    is_rel, score = check_paper_domain_relevance(query=query, paper_text=bio_text, paper_title="Oncogenic miRNA study")
    
    assert is_rel is False
    assert score == 0.0


def test_domain_relevance_gate_allows_quantum_for_quantum():
    query = "quantum resistant federated learning"
    quantum_text = "We propose a post-quantum lattice-based encryption scheme for privacy-preserving federated learning across decentralized clients."
    is_rel, score = check_paper_domain_relevance(query=query, paper_text=quantum_text, paper_title="Lattice Federated Learning")
    
    assert is_rel is True
    assert score >= 0.70


def test_domain_relevance_gate_blocks_bio_for_llm():
    query = "How does increasing model size affect LLM performance?"
    bio_text = "Insomnia and sleep deprivation impair hippocampal memory consolidation in murine models."
    is_rel, score = check_paper_domain_relevance(query=query, paper_text=bio_text, paper_title="Murine Sleep Patterns")
    
    assert is_rel is False
    assert score == 0.0


def test_evidence_mapper_thresholds():
    assert HIGH_RELEVANCE_THRESHOLD == 0.80
    assert SUPPORT_THRESHOLD == 0.70
    assert PARTIAL_THRESHOLD == 0.50

    # Test classify_relationship with unrelated text
    rel, sc = classify_relationship(
        claim_text="Quantum resistant federated learning achieves Byzantine fault tolerance.",
        passage_text="Patients with acute myocardial infarction were treated with beta blockers."
    )
    assert rel == "IRRELEVANT"
