"""PlannerAgent module for CiteGuard.

Translates arbitrary user research questions into structured scientific research plans
without inventing scientific evidence or predetermining outcomes:

USER RESEARCH QUESTION
         ↓
   PLANNER AGENT
         ↓
   RESEARCH PLAN (Objectives, Subquestions, Retrieval Queries, Evidence Requirements, Constraints)
         ↓
   PAPER RETRIEVAL (OpenAlex / Hybrid BM25 + FAISS + RRF)
         ↓
   CLAIM EXTRACTION
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.claim_generation.classifier import classify_question, QuestionType
from src.claim_generation.generator import parse_scientific_question, clean_phrase
from src.verification.providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class PlannerResult:
    """Structured and serializable research plan produced by PlannerAgent."""
    original_question: str
    research_objective: str
    question_type: str  # Canonical QuestionType value
    research_subquestions: List[str] = field(default_factory=list)
    retrieval_queries: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    expected_claim_types: List[str] = field(default_factory=list)
    population_constraints: List[str] = field(default_factory=list)
    domain_constraints: List[str] = field(default_factory=list)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    comparison_targets: Optional[List[str]] = None
    plan_confidence: float = 0.90
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerResult":
        """Instantiate PlannerResult from dictionary."""
        return cls(
            original_question=data.get("original_question", ""),
            research_objective=data.get("research_objective", ""),
            question_type=data.get("question_type", "SCIENTIFIC_HYPOTHESIS"),
            research_subquestions=data.get("research_subquestions", []),
            retrieval_queries=data.get("retrieval_queries", []),
            evidence_requirements=data.get("evidence_requirements", []),
            expected_claim_types=data.get("expected_claim_types", []),
            population_constraints=data.get("population_constraints", []),
            domain_constraints=data.get("domain_constraints", []),
            temporal_constraints=data.get("temporal_constraints", {}),
            comparison_targets=data.get("comparison_targets"),
            plan_confidence=float(data.get("plan_confidence", 0.90)),
            metadata=data.get("metadata", {})
        )


class PlannerAgent:
    """Adversarial research question planning agent.
    
    Converts unstructured research questions into structured, evidence-grounded
    retrieval and verification plans prior to literature retrieval and claim extraction.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider

    def plan(self, research_question: str) -> PlannerResult:
        """Construct an authoritative research plan from a research question."""
        if not research_question or not isinstance(research_question, str):
            raise ValueError("Research question must be a non-empty string.")

        q_clean = research_question.strip()
        q_lower = q_clean.lower()

        # Phase 3: Reuse and enrich existing question classification
        q_classification = classify_question(q_clean)
        q_type_enum = q_classification.get("question_type", QuestionType.SCIENTIFIC_HYPOTHESIS)
        q_type_str = q_type_enum.value if hasattr(q_type_enum, "value") else str(q_type_enum)

        # Parse scientific semantic components (domain, variables, relationship pattern)
        parsed_semantics = parse_scientific_question(q_clean)
        domain = parsed_semantics.get("domain", "general_science")
        var_x = parsed_semantics.get("var_x", "")
        var_y = parsed_semantics.get("var_y", "")
        clean_topic = q_classification.get("topic") or q_clean

        # 1. Synthesize Research Objective
        research_objective = self._formulate_objective(
            question=q_clean,
            q_type=q_type_str,
            var_x=var_x,
            var_y=var_y,
            topic=clean_topic
        )

        # 2. Extract Domain & Population Constraints
        domain_constraints = self._extract_domain_constraints(q_lower, domain)
        population_constraints = self._extract_population_constraints(q_lower, domain)

        # 3. Extract Temporal Constraints
        temporal_constraints = self._extract_temporal_constraints(q_classification, q_lower)

        # 4. Extract Comparison Targets (for comparative questions)
        comparison_targets = self._extract_comparison_targets(q_clean, q_type_str, var_x, var_y)

        # 5. Generate Meaningful Research Subquestions (Phase 4)
        research_subquestions = self._generate_subquestions(
            question=q_clean,
            q_type=q_type_str,
            var_x=var_x,
            var_y=var_y,
            topic=clean_topic,
            comparison_targets=comparison_targets
        )

        # 6. Generate Targeted Retrieval Queries (Phase 5)
        retrieval_queries = self._generate_retrieval_queries(
            question=q_clean,
            q_type=q_type_str,
            topic=clean_topic,
            var_x=var_x,
            var_y=var_y,
            domain=domain,
            comparison_targets=comparison_targets,
            temporal_constraints=temporal_constraints
        )

        # 7. Identify Evidence Requirements (Phase 6)
        evidence_requirements = self._determine_evidence_requirements(
            q_type=q_type_str,
            domain=domain,
            has_comparison=bool(comparison_targets),
            requires_doi=q_classification.get("requires_doi", False)
        )

        # 8. Expected Claim Types
        expected_claim_types = self._determine_expected_claim_types(q_type_str)

        # Build PlannerResult
        return PlannerResult(
            original_question=q_clean,
            research_objective=research_objective,
            question_type=q_type_str,
            research_subquestions=research_subquestions,
            retrieval_queries=retrieval_queries,
            evidence_requirements=evidence_requirements,
            expected_claim_types=expected_claim_types,
            population_constraints=population_constraints,
            domain_constraints=domain_constraints,
            temporal_constraints=temporal_constraints,
            comparison_targets=comparison_targets,
            plan_confidence=0.92,
            metadata={
                "parsed_semantics": parsed_semantics,
                "classification_parameters": q_classification,
                "planner_version": "2.0.0"
            }
        )

    def _formulate_objective(
        self,
        question: str,
        q_type: str,
        var_x: str,
        var_y: str,
        topic: str
    ) -> str:
        """Formulate a concise, evidence-neutral scientific research objective."""
        q_core = question.rstrip("?.")
        # Strip leading question auxiliaries
        q_statement = re.sub(
            r"^(?:does|do|can|is|are|how\s+does|how\s+do|what\s+is\s+the|what\s+are\s+the|which)\s+",
            "",
            q_core,
            flags=re.IGNORECASE
        ).strip()

        if q_type == "BIBLIOGRAPHIC_SEARCH":
            return f"Identify and retrieve high-impact, verified publications on '{topic}' with authentic citation and registry metadata."
        elif q_type == "COMPARATIVE_SCIENTIFIC":
            if var_x and var_y:
                return f"Evaluate and compare the relative mechanisms, performance, and trade-offs of {var_x} versus {var_y}."
            return f"Systematically compare relative efficacy, accuracy, and operational trade-offs for {q_statement}."
        elif q_type == "LITERATURE_REVIEW":
            return f"Synthesize established empirical evidence and state of the art concerning {topic}."
        elif var_x and var_y:
            return f"Evaluate the empirical relationship and causal mechanisms connecting {var_x} to {var_y}."
        else:
            return f"Investigate the empirical scientific evidence regarding {q_statement}."

    def _extract_domain_constraints(self, q_lower: str, domain: str) -> List[str]:
        """Extract primary and secondary domain constraints to guide retrieval topic gating."""
        constraints = [domain]
        if "quantum" in q_lower:
            constraints.append("quantum_information")
        if "federated" in q_lower or "distributed" in q_lower:
            constraints.append("distributed_systems")
        if "cancer" in q_lower or "tumor" in q_lower or "carcinoma" in q_lower:
            constraints.append("oncology")
        if "cardiovascular" in q_lower or "heart" in q_lower or "hypertension" in q_lower:
            constraints.append("cardiology")
        if "sleep" in q_lower or "cognit" in q_lower or "memory" in q_lower:
            constraints.append("neuroscience")
        if "llm" in q_lower or "transformer" in q_lower or "language model" in q_lower:
            constraints.append("natural_language_processing")
        if "solar" in q_lower or "photovoltaic" in q_lower or "perovskite" in q_lower:
            constraints.append("photovoltaics_energy")
        return list(dict.fromkeys(constraints))

    def _extract_population_constraints(self, q_lower: str, domain: str) -> List[str]:
        """Identify human, animal, in vitro, or computational model constraints."""
        pop = []
        if any(w in q_lower for w in ["human", "patient", "clinical", "cohort", "adult", "children", "elderly"]):
            pop.append("human_clinical_cohorts")
        if any(w in q_lower for w in ["mice", "mouse", "rodent", "animal", "rat", "in vivo"]):
            pop.append("in_vivo_rodent_models")
        if any(w in q_lower for w in ["cell", "cell line", "in vitro", "culture", "assay"]):
            pop.append("in_vitro_cellular_systems")
        if domain == "computer_science" or any(w in q_lower for w in ["benchmark", "model", "algorithm", "dataset"]):
            pop.append("computational_benchmark_corpora")
        
        if not pop:
            if domain == "biomedical":
                pop.append("general_biological_or_clinical_systems")
            else:
                pop.append("peer_reviewed_experimental_settings")
        return pop

    def _extract_temporal_constraints(self, q_class: Dict[str, Any], q_lower: str) -> Dict[str, Any]:
        """Identify target years, recency priorities, and chronological boundaries."""
        t_constraints: Dict[str, Any] = {
            "target_year": q_class.get("target_year"),
            "recency_preferred": False,
            "max_age_years": 15
        }
        if q_class.get("target_year"):
            t_constraints["recency_preferred"] = True
            t_constraints["constraint_type"] = "exact_or_recent_to_year"
        elif any(w in q_lower for w in ["recent", "latest", "modern", "state of the art", "current"]):
            t_constraints["recency_preferred"] = True
            t_constraints["max_age_years"] = 5
            t_constraints["constraint_type"] = "contemporary_sota"
        elif any(w in q_lower for w in ["llm", "large language model", "transformer", "quantum federated"]):
            t_constraints["recency_preferred"] = True
            t_constraints["max_age_years"] = 6
            t_constraints["constraint_type"] = "rapidly_evolving_discipline"
        else:
            t_constraints["constraint_type"] = "standard_peer_reviewed_window"
        return t_constraints

    def _extract_comparison_targets(
        self,
        question: str,
        q_type: str,
        var_x: str,
        var_y: str
    ) -> Optional[List[str]]:
        """Extract explicit baseline vs candidate comparison targets."""
        if q_type != "COMPARATIVE_SCIENTIFIC":
            return None

        q_lower = question.lower()
        # Look for explicit "versus" or "vs"
        vs_match = re.search(r"([a-zA-Z0-9_\-\s]+?)\s+(?:versus|vs\.?|compared to)\s+([a-zA-Z0-9_\-\s]+)", question, re.IGNORECASE)
        if vs_match:
            t1 = clean_phrase(vs_match.group(1))
            t2 = clean_phrase(vs_match.group(2))
            return [t1, t2]

        if "model size" in q_lower and ("accuracy" in q_lower or "computational cost" in q_lower):
            return ["smaller parameter configurations", "scaled parameter configurations"]

        if var_x and var_y:
            return [var_x, var_y]

        return ["baseline configuration", "proposed / alternative configuration"]

    def _generate_subquestions(
        self,
        question: str,
        q_type: str,
        var_x: str,
        var_y: str,
        topic: str,
        comparison_targets: Optional[List[str]] = None
    ) -> List[str]:
        """Generate meaningful research subquestions decomposing the inquiry."""
        subqs: List[str] = []

        if q_type == "BIBLIOGRAPHIC_SEARCH":
            subqs.append(f"What publications on {topic} have garnered the highest citation impact?")
            subqs.append("Which candidate papers align with specified publication year and venue criteria?")
            subqs.append("Can authentic digital object identifiers (DOIs) and authors be verified for each paper?")
            subqs.append("What are the primary methodological contributions established in these influential works?")
            return subqs

        if q_type == "COMPARATIVE_SCIENTIFIC":
            t1 = comparison_targets[0] if comparison_targets and len(comparison_targets) > 0 else "baseline"
            t2 = comparison_targets[1] if comparison_targets and len(comparison_targets) > 1 else "target"
            subqs.append(f"What empirical outcomes or benchmark tasks have evaluated {t1} against {t2}?")
            subqs.append("Under what experimental or operational conditions was the comparison conducted?")
            subqs.append(f"What quantitative metrics (e.g. accuracy, latency, efficiency, throughput) distinguish {t1} and {t2}?")
            subqs.append("What trade-offs between performance and computational or biological cost are reported?")
            subqs.append("Are there specific scenarios or configurations where the comparative advantage reverses?")
            subqs.append("Are there contradictory or statistically insignificant comparative findings in the literature?")
            return subqs

        # Causal / Hypothesis / General Scientific Subquestions
        x_label = var_x or "the primary variable"
        y_label = var_y or "the target outcome"

        subqs.append(f"What specific outcomes or biomarkers representing {y_label} have been empirically studied?")
        subqs.append(f"What study populations, experimental models, or test systems were evaluated for {x_label}?")
        subqs.append(f"What dosage, intensity, model parameterization, or duration of {x_label} were tested?")
        subqs.append("Is the published evidence primarily observational, randomized controlled, or mechanistic in vitro?")
        subqs.append(f"What effect direction (positive, inverse, null) is consistently reported between {x_label} and {y_label}?")
        subqs.append("Are there documented contradictory, null, or statistically disputed findings in peer-reviewed literature?")

        return subqs

    def _generate_retrieval_queries(
        self,
        question: str,
        q_type: str,
        topic: str,
        var_x: str,
        var_y: str,
        domain: str,
        comparison_targets: Optional[List[str]],
        temporal_constraints: Dict[str, Any]
    ) -> List[str]:
        """Generate targeted retrieval queries optimized for OpenAlex, BM25, and FAISS."""
        queries: List[str] = []

        # 1. Cleaned primary query (strip punctuation and question syntax)
        clean_q = re.sub(r"[?.,!:]+", "", question).strip()
        queries.append(clean_q)

        # Clean topic query
        t_clean = re.sub(r"[?.,!:]+", "", topic).strip()
        if t_clean and t_clean.lower() != clean_q.lower():
            queries.append(t_clean)

        # 2. Semantic variable combinations
        if var_x and var_y:
            queries.append(f"{var_x} {var_y}")
            queries.append(f"{var_x} {var_y} experimental evidence")
            queries.append(f"{var_x} {var_y} mechanism")
            queries.append(f"{var_x} {var_y} systematic review")

        # 3. Comparative queries
        if q_type == "COMPARATIVE_SCIENTIFIC" and comparison_targets:
            c1, c2 = comparison_targets[0], comparison_targets[1]
            queries.append(f"{c1} vs {c2} benchmark comparison")
            queries.append(f"{c1} {c2} trade-off performance")
            if "model size" in question.lower():
                queries.append("large language models parameter scaling accuracy compute cost")
                queries.append("transformer scaling laws computational trade-offs")

        # 4. Bibliographic query enrichment
        if q_type == "BIBLIOGRAPHIC_SEARCH":
            target_yr = temporal_constraints.get("target_year")
            if target_yr:
                queries.append(f"{t_clean} {target_yr}")
                queries.append(f"{t_clean} most cited {target_yr}")
            queries.append(f"{t_clean} highly cited survey")
            queries.append(f"{t_clean} review paper")

        # 5. Domain-specific query additions
        if domain == "biomedical":
            queries.append(f"{t_clean} clinical trial cohort study")
        elif domain == "computer_science":
            queries.append(f"{t_clean} empirical evaluation benchmark")

        # Deduplicate while preserving order
        deduped: List[str] = []
        for q in queries:
            q_norm = " ".join(q.split())
            if q_norm and q_norm.lower() not in [d.lower() for d in deduped]:
                deduped.append(q_norm)

        return deduped[:6]

    def _determine_evidence_requirements(
        self,
        q_type: str,
        domain: str,
        has_comparison: bool,
        requires_doi: bool
    ) -> List[str]:
        """Establish rigorous evidence verification requirements."""
        reqs = [
            "Study Design: Explicit empirical experimental design, clinical trial, or benchmark corpus",
            "Literature Grounding: Published in peer-reviewed scientific venue with authentic index registry entry"
        ]

        if q_type == "BIBLIOGRAPHIC_SEARCH" or requires_doi:
            reqs.append("Bibliographic Provenance: Exact registered DOI and verified author attribution")
            reqs.append("Citation Metric: Verified cited-by count registered in OpenAlex / CrossRef index")
            reqs.append("Temporal Match: Confirmed publication year conforming to target constraints")
            return reqs

        if q_type == "COMPARATIVE_SCIENTIFIC" or has_comparison:
            reqs.append("Comparison Conditions: Standardized baseline vs candidate benchmark environment")
            reqs.append("Quantitative Metrics: Explicit numerical measurements for accuracy, latency, or effect size")
            reqs.append("Trade-off Characterization: Clear documentation of cost-benefit relationship across configurations")

        if domain == "biomedical":
            reqs.append("Biological Scope: Defined organism, sample size, or cellular model system")
            reqs.append("Causal Attribution: Distinguish observational association from mechanistic causation")
        else:
            reqs.append("Reproducibility: Documented hyperparameter, statistical, or evaluation protocol")

        reqs.append("Contradiction Scrutiny: Verification against conflicting or non-replicable findings")
        return reqs

    def _determine_expected_claim_types(self, q_type: str) -> List[str]:
        """Determine downstream claim classification categories."""
        if q_type == "BIBLIOGRAPHIC_SEARCH":
            return ["bibliographic_citation", "publication_metadata", "citation_count_ranking"]
        elif q_type == "COMPARATIVE_SCIENTIFIC":
            return ["comparative_finding", "performance_tradeoff", "empirical_assertion"]
        else:
            return ["empirical_assertion", "causal_hypothesis", "statistical_finding"]
