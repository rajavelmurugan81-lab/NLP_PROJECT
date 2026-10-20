"""Comprehensive Unit Tests for CiteGuard PlannerAgent."""

import pytest
import json
from src.planner.planner_agent import PlannerAgent, PlannerResult
from src.claim_generation.classifier import QuestionType, classify_question
from src.research_pipeline import ResearchPipeline


def test_planner_general_scientific_question():
    """Verify planner handles general scientific questions with proper domain and objectives."""
    planner = PlannerAgent()
    q = "What factors influence solar cell efficiency?"
    plan = planner.plan(q)

    assert plan.original_question == q
    assert "solar cell efficiency" in plan.research_objective.lower()
    assert plan.plan_confidence >= 0.85
    assert len(plan.research_subquestions) >= 4
    assert len(plan.retrieval_queries) >= 2
    assert "photovoltaics_energy" in plan.domain_constraints or "physics_material" in plan.domain_constraints
    assert "empirical_assertion" in plan.expected_claim_types


def test_planner_comparative_question():
    """Verify planner detects comparative structure, targets, and trade-off requirements."""
    planner = PlannerAgent()
    q = "How does model size affect the accuracy and computational cost of large language models?"
    plan = planner.plan(q)

    assert plan.question_type == QuestionType.COMPARATIVE_SCIENTIFIC.value
    assert plan.comparison_targets is not None
    assert len(plan.comparison_targets) >= 2
    assert any("trade-off" in req.lower() or "quantitative" in req.lower() for req in plan.evidence_requirements)
    assert any("trade-offs" in sq.lower() or "metrics" in sq.lower() for sq in plan.research_subquestions)
    assert any("accuracy" in q_term for q_term in plan.retrieval_queries)


def test_planner_hypothesis_question():
    """Verify planner handles causal / hypothesis questions with outcome and population decomposition."""
    planner = PlannerAgent()
    q = "Does regular exercise reduce cardiovascular disease risk?"
    plan = planner.plan(q)

    assert plan.question_type == QuestionType.SCIENTIFIC_HYPOTHESIS.value
    assert "exercise" in plan.research_objective.lower()
    assert "cardiovascular" in plan.research_objective.lower()
    assert "cardiology" in plan.domain_constraints or "biomedical" in plan.domain_constraints
    
    # Verify subquestions cover outcomes, populations, intensity, causality, and contradiction
    subqs_text = " ".join(plan.research_subquestions).lower()
    assert "outcome" in subqs_text or "risk" in subqs_text
    assert "population" in subqs_text or "cohort" in subqs_text
    assert "observational" in subqs_text or "causation" in subqs_text or "mechanistic" in subqs_text
    assert "contradictory" in subqs_text or "disputed" in subqs_text


def test_planner_bibliographic_question():
    """Verify planner extracts bibliographic constraints, DOIs, target year, and ranking metrics."""
    planner = PlannerAgent()
    q = "What are the 10 most cited papers published in 2026 on quantum federated learning and provide their exact DOI?"
    plan = planner.plan(q)

    assert plan.question_type == QuestionType.BIBLIOGRAPHIC_SEARCH.value
    assert plan.temporal_constraints.get("target_year") == 2026
    assert plan.temporal_constraints.get("recency_preferred") is True
    assert any("doi" in req.lower() for req in plan.evidence_requirements)
    assert any("citation metric" in req.lower() or "cited-by" in req.lower() for req in plan.evidence_requirements)
    assert "quantum_information" in plan.domain_constraints
    assert "distributed_systems" in plan.domain_constraints


def test_planner_multi_subquestion_generation():
    """Verify planner produces 4-6 diverse, informative subquestions without trivial repetition."""
    planner = PlannerAgent()
    questions = [
        "Does miRNA expression play a role in cancer progression?",
        "How does CRISPR-Cas9 enable targeted DNA cleavage?",
        "What is the relationship between sleep duration and cognitive performance?"
    ]
    for q in questions:
        plan = planner.plan(q)
        assert len(plan.research_subquestions) >= 4
        # Verify no duplicate subquestions
        assert len(plan.research_subquestions) == len(set(plan.research_subquestions))
        for sq in plan.research_subquestions:
            assert len(sq.strip()) > 15
            assert sq.endswith("?")


def test_planner_retrieval_query_generation():
    """Verify planner generates multiple targeted retrieval queries suitable for search indices."""
    planner = PlannerAgent()
    q = "Does dietary sodium cause hypertension in adults?"
    plan = planner.plan(q)

    assert len(plan.retrieval_queries) >= 3
    # Queries should not contain punctuation or awkward question framing
    for q_term in plan.retrieval_queries:
        assert "?" not in q_term
        assert ":" not in q_term
        assert len(q_term.split()) >= 2
    # Topic words should be present
    all_queries_text = " ".join(plan.retrieval_queries).lower()
    assert "sodium" in all_queries_text
    assert "hypertension" in all_queries_text


def test_planner_domain_constraint_extraction():
    """Verify domain constraints reflect scientific fields accurately."""
    planner = PlannerAgent()
    
    p_med = planner.plan("Does microRNA-21 downregulate PTEN in glioblastoma?")
    assert "biomedical" in p_med.domain_constraints or "oncology" in p_med.domain_constraints

    p_cs = planner.plan("How do transformer scaling laws predict inference latency in large language models?")
    assert "computer_science" in p_cs.domain_constraints or "natural_language_processing" in p_cs.domain_constraints


def test_planner_temporal_constraint_extraction():
    """Verify temporal constraints correctly extract target years and recency requirements."""
    planner = PlannerAgent()
    
    # Specific year
    p_year = planner.plan("Find the top machine learning papers published in 2024.")
    assert p_year.temporal_constraints.get("target_year") == 2024

    # Rapidly evolving LLM topic
    p_llm = planner.plan("What are modern LLM parameter scaling limits?")
    assert p_llm.temporal_constraints.get("recency_preferred") is True
    assert p_llm.temporal_constraints.get("max_age_years") <= 6


def test_planner_no_fabricated_evidence():
    """Verify planner produces strictly research criteria and zero fabricated evidence."""
    planner = PlannerAgent()
    q = "Does unverified compound Z cure advanced Alzheimer's disease?"
    plan = planner.plan(q)

    # Must only contain criteria, constraints, and requirements
    plan_dict = plan.to_dict()
    assert "evidence_requirements" in plan_dict
    assert "original_question" in plan_dict
    assert "research_objective" in plan_dict
    # Ensure no fabricated text passages or mock citations exist in plan output
    assert "passages" not in plan_dict
    assert "citations" not in plan_dict
    assert "verdict" not in plan_dict


def test_planner_serialization():
    """Verify PlannerResult serializes cleanly to dict and JSON and roundtrips via from_dict."""
    planner = PlannerAgent()
    q = "How does CRISPR-Cas9 achieve sequence-specific DNA cleavage?"
    plan = planner.plan(q)

    d = plan.to_dict()
    assert isinstance(d, dict)
    # Check JSON serializability
    json_str = json.dumps(d)
    assert len(json_str) > 100

    # Roundtrip from_dict
    restored = PlannerResult.from_dict(json.loads(json_str))
    assert restored.original_question == plan.original_question
    assert restored.research_objective == plan.research_objective
    assert restored.question_type == plan.question_type
    assert restored.research_subquestions == plan.research_subquestions
    assert restored.retrieval_queries == plan.retrieval_queries
    assert restored.evidence_requirements == plan.evidence_requirements
    assert restored.domain_constraints == plan.domain_constraints


def test_planner_existing_question_classifier_compatibility():
    """Verify planner preserves and enriches existing classify_question output."""
    planner = PlannerAgent()
    q = "What are the 5 top cited papers on sleep and cognition published in 2026?"
    
    existing_class = classify_question(q)
    plan = planner.plan(q)

    assert plan.question_type == existing_class["question_type"].value
    assert plan.temporal_constraints.get("target_year") == existing_class.get("target_year")
    assert "classification_parameters" in plan.metadata


def test_planner_integration_with_research_pipeline():
    """Verify ResearchPipeline runs PlannerAgent and embeds research_plan into results and metadata."""
    pipeline = ResearchPipeline(provider_type="mock")
    q = "Does regular exercise reduce cardiovascular disease risk?"
    
    res = pipeline.run_pipeline(research_question=q, top_k=3, verbose=False)

    assert "research_plan" in res
    plan_dict = res["research_plan"]
    assert plan_dict["original_question"] == q
    assert len(plan_dict["research_subquestions"]) >= 4
    assert len(plan_dict["retrieval_queries"]) >= 2

    # Check research question container metadata
    rq = res["research_question"]
    assert "research_plan" in rq.metadata
    assert len(rq.sub_questions) >= 4
