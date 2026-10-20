"""Tests for Scientific Question Classifier and Parameter Extraction."""

import pytest
from src.claim_generation.classifier import classify_question, QuestionType


def test_classify_scientific_hypothesis():
    q = "Does sleep improve cognitive performance?"
    res = classify_question(q)
    assert res["question_type"] == QuestionType.SCIENTIFIC_HYPOTHESIS
    assert "sleep" in res["topic"].lower()


def test_classify_comparative_scientific():
    q = "How does increasing model size affect LLM performance?"
    res = classify_question(q)
    assert res["question_type"] == QuestionType.COMPARATIVE_SCIENTIFIC


def test_classify_bibliographic_search_query_5():
    q = "what are the 10 most cited papers published in 2026 on quantum resistant federated learning and provide their exact DOI,citation count and experimental results?"
    res = classify_question(q)
    assert res["question_type"] == QuestionType.BIBLIOGRAPHIC_SEARCH
    assert res["target_count"] == 10
    assert res["target_year"] == 2026
    assert res["ranking_metric"] == "cited_by_count"
    assert res["requires_doi"] is True
    assert res["requires_citations"] is True
    assert res["requires_experimental_results"] is True
    assert "quantum resistant federated learning" in res["topic"].lower()


def test_classify_literature_review():
    q = "Can you provide a comprehensive literature review and overview of CRISPR-Cas9?"
    res = classify_question(q)
    assert res["question_type"] == QuestionType.LITERATURE_REVIEW


def test_classify_metadata_request():
    q = "What is the DOI for the original Transformer paper?"
    res = classify_question(q)
    assert res["question_type"] == QuestionType.METADATA_REQUEST
