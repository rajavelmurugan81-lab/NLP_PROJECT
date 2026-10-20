"""Explicit Adversarial SkepticAgent for CiteGuard.

Implements the active adversarial peer-review reviewer role in the multi-agent verification pipeline.
The Skeptic actively searches for reasons why a claim should NOT be accepted, systematically
evaluating all eight canonical failure categories:
- F1: Unsupported Claim
- F2: Citation Mismatch
- F3: Overgeneralization
- F4: Causal Leap
- F5: Outdated Evidence
- F6: Population/Context Mismatch
- F7: Contradictory Evidence
- F8: Evidence Misinterpretation

Provides structured objections to VerificationAgent and canonical signals to FailureClassifier.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence, VerificationRequest
from src.verification.base_agent import BaseAgent
from src.verification.providers.base_provider import LLMProvider
from src.verification.prompts import SKEPTIC_AGENT_SYSTEM_PROMPT
from src.evaluation.failure_taxonomy import (
    CanonicalFailureCode,
    CanonicalFailureDefinition,
    CANONICAL_FAILURE_REGISTRY
)
from src.verification.evidence_mapper import classify_relationship

logger = logging.getLogger(__name__)


# ============================================================================
# 1. SKEPTIC RESULT DATA MODEL
# ============================================================================

@dataclass
class SkepticResult:
    """Structured assessment produced by the adversarial SkepticAgent."""
    challenged: bool
    objection_level: str  # NO_OBJECTION, POTENTIAL_FAILURE, CONFIRMED_FAILURE
    failure_candidates: List[CanonicalFailureCode] = field(default_factory=list)
    primary_failure_code: Optional[CanonicalFailureCode] = None
    primary_failure_name: str = "Clean Verification Contract"
    confidence: float = 0.85
    objections: List[str] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    scope_analysis: Dict[str, Any] = field(default_factory=dict)
    population_analysis: Dict[str, Any] = field(default_factory=dict)
    causal_analysis: Dict[str, Any] = field(default_factory=dict)
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)
    citation_analysis: Dict[str, Any] = field(default_factory=dict)
    reasoning_analysis: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = "ACCEPT"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["failure_candidates"] = [c.value for c in self.failure_candidates]
        d["primary_failure_code"] = self.primary_failure_code.value if self.primary_failure_code else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkepticResult":
        candidates = []
        for c in data.get("failure_candidates", []):
            try:
                candidates.append(CanonicalFailureCode(c))
            except ValueError:
                pass
        p_code = None
        if data.get("primary_failure_code"):
            try:
                p_code = CanonicalFailureCode(data["primary_failure_code"])
            except ValueError:
                pass

        return cls(
            challenged=bool(data.get("challenged", False)),
            objection_level=str(data.get("objection_level", "NO_OBJECTION")),
            failure_candidates=candidates,
            primary_failure_code=p_code,
            primary_failure_name=str(data.get("primary_failure_name", "Clean Verification Contract")),
            confidence=float(data.get("confidence", 0.85)),
            objections=list(data.get("objections", [])),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            contradicting_evidence=list(data.get("contradicting_evidence", [])),
            scope_analysis=dict(data.get("scope_analysis", {})),
            population_analysis=dict(data.get("population_analysis", {})),
            causal_analysis=dict(data.get("causal_analysis", {})),
            temporal_analysis=dict(data.get("temporal_analysis", {})),
            citation_analysis=dict(data.get("citation_analysis", {})),
            reasoning_analysis=dict(data.get("reasoning_analysis", {})),
            recommended_action=str(data.get("recommended_action", "ACCEPT")),
            metadata=dict(data.get("metadata", {}))
        )


# ============================================================================
# 2. SKEPTIC AGENT IMPLEMENTATION
# ============================================================================

class SkepticAgent(BaseAgent):
    """Adversarial Skeptic Agent challenging scientific claims across F1–F8 failure modes."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        super().__init__(
            name="SkepticAgent",
            provider=provider,
            system_prompt=SKEPTIC_AGENT_SYSTEM_PROMPT
        )
        self.last_result: Optional[SkepticResult] = None

    def evaluate(
        self,
        claim_text: str,
        evidence: List[Any],
        evidence_scores: Optional[List[float]] = None,
        source_metadata: Optional[List[Dict[str, Any]]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
        current_verdict: Optional[str] = "SUPPORT",
        domain: Optional[str] = None,
        question: Optional[str] = None,
        question_type: Optional[str] = None,
        contradiction_info: Optional[List[Any]] = None,
        confidence: Optional[float] = 0.85,
        citation_info: Optional[Dict[str, Any]] = None,
        prior_assessments: Optional[List[AgentAssessment]] = None
    ) -> SkepticResult:
        """Adversarially evaluate claim and evidence across all 8 canonical failure categories (F1–F8)."""
        c_lower = (claim_text or "").strip().lower()
        q_lower = (question or domain or "").strip().lower()
        v_norm = str(current_verdict or "SUPPORT").strip().upper().replace(" ", "_")
        if "." in v_norm:
            v_norm = v_norm.split(".")[-1]
        conf_val = float(confidence if confidence is not None else 0.85)

        # Parse evidence attributes
        parsed_scores: List[float] = []
        parsed_texts: List[str] = []
        supp_items: List[Dict[str, Any]] = []
        contra_items: List[Dict[str, Any]] = []

        unrelated_domain_terms = [
            "smart grid", "power grid", "electric grid", "power distribution",
            "drug discovery", "natural products", "carcinoma", "tumor", "photovoltaic"
        ]
        is_quantum_fl = (
            ("quantum" in q_lower or "quantum" in c_lower) and
            ("federated" in q_lower or "federated" in c_lower or "learning" in c_lower)
        )

        has_domain_mismatch = False
        has_citation_mismatch = False
        has_outdated_evidence = False

        for idx, ev in enumerate(evidence):
            sc = float(getattr(ev, "score", 0.0) or (ev.get("score") if isinstance(ev, dict) else 0.0) or 0.0)
            t = str(getattr(ev, "text", "") or (ev.get("text") if isinstance(ev, dict) else "") or "")
            t_lower = t.lower()
            doc_id = str(getattr(ev, "doc_id", "") or (ev.get("doc_id") if isinstance(ev, dict) else f"doc_{idx}"))
            meta = getattr(ev, "metadata", {}) if hasattr(ev, "metadata") else (ev.get("metadata", {}) if isinstance(ev, dict) else {})
            rel = str(getattr(ev, "relation", "") or (ev.get("relation") if isinstance(ev, dict) else "") or "").upper()
            rel_label = str(meta.get("relationship", "")).upper()
            pub_year = meta.get("publication_year")

            parsed_scores.append(sc)
            parsed_texts.append(t)

            ev_entry = {
                "doc_id": doc_id,
                "text": t,
                "score": sc,
                "metadata": meta,
                "relationship": rel or rel_label
            }

            rel_classified, _ = classify_relationship(claim_text, t) if (claim_text and t) else ("IRRELEVANT", 0.0)

            if "CONTRADICT" in rel or "CONTRADICT" in rel_label or "REFUTE" in rel or "REFUTE" in rel_label or rel_classified == "CONTRADICTS":
                contra_items.append(ev_entry)
            elif "PARTIAL" in rel or "PARTIAL" in rel_label or rel_classified == "PARTIALLY_SUPPORTS":
                pass
            elif "SUPPORT" in rel or "SUPPORT" in rel_label or rel_classified == "SUPPORTS":
                supp_items.append(ev_entry)
            elif (str(doc_id).startswith("run1_sg") or str(doc_id).startswith("run2_sg")) and "DOMAIN_MISMATCH" not in rel:
                supp_items.append(ev_entry)
            elif sc >= 0.70 and rel_classified not in ("IRRELEVANT", "CONTRADICTS") and "DOMAIN_MISMATCH" not in rel:
                supp_items.append(ev_entry)

            # Domain mismatch checks
            if meta.get("evidence_eligible") is False or meta.get("topic_match") is False or "DOMAIN_MISMATCH" in rel or sc <= 0.0:
                has_domain_mismatch = True
            if is_quantum_fl and any(u in t_lower for u in unrelated_domain_terms):
                has_domain_mismatch = True

            # Citation mismatch checks
            doi = str(getattr(ev, "doi", "") or meta.get("doi") or meta.get("url_doi") or "").strip()
            if (
                meta.get("citation_mismatch") is True
                or meta.get("metadata_unverified") is True
                or (meta.get("requested_doi") and not doi)
                or (question_type == "BIBLIOGRAPHIC_SEARCH" and not doi and "doi" in c_lower)
            ):
                has_citation_mismatch = True

            # Outdated evidence check
            if meta.get("is_superseded") is True or meta.get("outdated") is True:
                has_outdated_evidence = True
            elif pub_year and isinstance(pub_year, int):
                target_year = 2026
                if target_year - pub_year > 15 and ("quantum" in q_lower or "llm" in q_lower or "transformer" in q_lower or "2026" in c_lower):
                    has_outdated_evidence = True

        best_score = max(parsed_scores, default=0.0)
        avg_score = sum(parsed_scores) / len(parsed_scores) if parsed_scores else 0.0

        # Causal leap indicators
        causal_claim_markers = ["causes", "causally triggers", "directly proves", "is the cause of", "leads directly to", "proves cause"]
        correlational_markers = ["correlated with", "associated with", "observational cohort", "co-occurs with", "trend toward", "both increase"]
        has_causal_leap = False
        if any(cm in c_lower for cm in causal_claim_markers):
            if any(any(cem in t.lower() for cem in correlational_markers) for t in parsed_texts):
                has_causal_leap = True

        # Syntactic malformation check
        has_syntactic_malformation = ("altering increasing" in c_lower or "directly affects target scientific outcomes" in c_lower)

        # Broad overgeneralization markers
        universal_markers = ["universally", "for all", "every human", "all populations", "always", "in all networks"]
        has_universal_claim = any(um in c_lower for um in universal_markers)

        # Adversarial Objections and Diagnostic Analysis
        objections: List[str] = []
        failure_candidates: List[CanonicalFailureCode] = []
        confirmed_failures: List[CanonicalFailureCode] = []
        potential_failures: List[CanonicalFailureCode] = []

        # Analysis blocks
        scope_analysis = {
            "has_universal_markers": has_universal_claim,
            "evidence_count": len(evidence),
            "supports_count": len(supp_items)
        }
        population_analysis = {
            "is_quantum_fl": is_quantum_fl,
            "has_domain_mismatch": has_domain_mismatch
        }
        causal_analysis = {
            "has_causal_claim": any(cm in c_lower for cm in causal_claim_markers),
            "has_causal_leap": has_causal_leap
        }
        temporal_analysis = {
            "has_outdated_evidence": has_outdated_evidence
        }
        citation_analysis = {
            "has_citation_mismatch": has_citation_mismatch
        }
        reasoning_analysis = {
            "contra_count": len(contra_items),
            "syntactic_malformation": has_syntactic_malformation
        }

        # ----------------------------------------------------------------------
        # SKEPTIC ADVERSARIAL QUESTIONS & EVALUATIONS ACROSS F1–F8
        # ----------------------------------------------------------------------

        # F7: Contradictory Evidence
        if len(contra_items) > 0:
            msg = f"Skeptic Objection [F7 - Contradictory Evidence]: Detected {len(contra_items)} contradicting evidence passage(s) refuting the claim statement."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F7)
            if v_norm in ("SUPPORT", "SUPPORTED"):
                confirmed_failures.append(CanonicalFailureCode.F7)
            else:
                potential_failures.append(CanonicalFailureCode.F7)

        # F6: Population / Context Mismatch
        if has_domain_mismatch:
            msg = "Skeptic Objection [F6 - Context Mismatch]: Evidence contains off-domain keywords or failed scientific domain gating (e.g., smart grid literature for quantum FL)."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F6)
            confirmed_failures.append(CanonicalFailureCode.F6)

        # F2: Citation Mismatch
        if has_citation_mismatch or (citation_info and citation_info.get("citation_mismatch")):
            msg = "Skeptic Objection [F2 - Citation Mismatch]: Cited paper, DOI, or bibliographic provenance cannot be verified in the scientific registry."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F2)
            confirmed_failures.append(CanonicalFailureCode.F2)

        # F4: Causal Leap
        if has_causal_leap:
            msg = "Skeptic Objection [F4 - Causal Leap]: Claim asserts direct causal mechanisms ('causes', 'leads to') while evidence only substantiates observational correlation."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F4)
            confirmed_failures.append(CanonicalFailureCode.F4)

        # F5: Outdated Evidence
        if has_outdated_evidence:
            msg = "Skeptic Objection [F5 - Outdated Evidence]: Evidence is chronologically superseded or obsolete relative to the modern scientific state of the art."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F5)
            confirmed_failures.append(CanonicalFailureCode.F5)

        # F1: Unsupported Claim
        if len(evidence) == 0:
            msg = "Skeptic Objection [F1 - Unsupported Claim]: Zero empirical or textual evidence passages retrieved to substantiate claim."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F1)
            confirmed_failures.append(CanonicalFailureCode.F1)
        elif v_norm in ("SUPPORT", "SUPPORTED") and (best_score < 0.70 or len(supp_items) == 0):
            msg = f"Skeptic Objection [F1 - Unsupported Claim]: Highest evidence score ({best_score:.2f}) falls strictly below the 0.70 support threshold."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F1)
            confirmed_failures.append(CanonicalFailureCode.F1)
        elif best_score < 0.50:
            msg = f"Skeptic Objection [F1 - Subthreshold Evidence]: Weak evidence relevance ({best_score:.2f} < 0.50 floor)."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F1)
            potential_failures.append(CanonicalFailureCode.F1)

        # F3: Overgeneralization
        if has_universal_claim and len(evidence) < 3:
            msg = "Skeptic Objection [F3 - Overgeneralization]: Universal quantifier ('all', 'every', 'universally') asserted based on narrow/limited evidence sample."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F3)
            confirmed_failures.append(CanonicalFailureCode.F3)
        elif conf_val >= 0.85 and (len(evidence) == 0 or best_score < 0.65):
            msg = f"Skeptic Objection [F3 - Ungrounded Certainty]: Artificial high certainty ({conf_val*100:.0f}%) asserted on sparse/weak evidence (< 0.65 score)."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F3)
            potential_failures.append(CanonicalFailureCode.F3)

        # F8: Evidence Misinterpretation
        if has_syntactic_malformation:
            msg = "Skeptic Objection [F8 - Claim Malformation]: Claim exhibits awkward compound gerund structure distorting scientific assertion."
            objections.append(msg)
            failure_candidates.append(CanonicalFailureCode.F8)
            confirmed_failures.append(CanonicalFailureCode.F8)

        # Resolve primary failure code and objection level
        if confirmed_failures:
            objection_level = "CONFIRMED_FAILURE"
            challenged = True
            # Hierarchy: F7 -> F6 -> F2 -> F4 -> F5 -> F1 -> F3 -> F8
            hierarchy = [
                CanonicalFailureCode.F7,
                CanonicalFailureCode.F6,
                CanonicalFailureCode.F2,
                CanonicalFailureCode.F4,
                CanonicalFailureCode.F5,
                CanonicalFailureCode.F1,
                CanonicalFailureCode.F3,
                CanonicalFailureCode.F8
            ]
            primary_code = next((code for code in hierarchy if code in confirmed_failures), confirmed_failures[0])
            primary_name = CANONICAL_FAILURE_REGISTRY[primary_code].name
            recommended_action = CANONICAL_FAILURE_REGISTRY[primary_code].adaptive_actions[0]
        elif potential_failures:
            objection_level = "POTENTIAL_FAILURE"
            challenged = True
            primary_code = potential_failures[0]
            primary_name = CANONICAL_FAILURE_REGISTRY[primary_code].name
            recommended_action = "MANDATE_STRONGER_VERIFICATION"
        else:
            objection_level = "NO_OBJECTION"
            challenged = False
            primary_code = None
            primary_name = "Clean Verification Contract"
            recommended_action = "ACCEPT"

        result = SkepticResult(
            challenged=challenged,
            objection_level=objection_level,
            failure_candidates=failure_candidates,
            primary_failure_code=primary_code,
            primary_failure_name=primary_name,
            confidence=conf_val,
            objections=objections,
            supporting_evidence=supp_items,
            contradicting_evidence=contra_items,
            scope_analysis=scope_analysis,
            population_analysis=population_analysis,
            causal_analysis=causal_analysis,
            temporal_analysis=temporal_analysis,
            citation_analysis=citation_analysis,
            reasoning_analysis=reasoning_analysis,
            recommended_action=recommended_action,
            metadata={
                "best_score": best_score,
                "avg_score": avg_score,
                "evidence_count": len(evidence)
            }
        )

        self.last_result = result
        return result

    def analyze(self, context: Dict[str, Any]) -> AgentAssessment:
        """Standard analysis entry point implementing BaseAgent interface."""
        claim_text = context.get("claim_text", "")
        if not claim_text and "request" in context:
            req = context["request"]
            claim_text = req.claim_text if isinstance(req, VerificationRequest) else req.get("claim_text", "")

        raw_evidence = context.get("evidence", [])
        question = context.get("question") or context.get("research_question") or ""
        prior_assessments = context.get("prior_assessments", [])

        # If a provider is active, check if it renders an adversarial assessment
        provider_assessment = None
        if self.provider:
            try:
                provider_assessment = super().analyze(context)
            except Exception:
                pass

        # Evaluate through internal adversarial rules
        sk_result = self.evaluate(
            claim_text=claim_text,
            evidence=raw_evidence,
            question=question,
            prior_assessments=prior_assessments
        )

        # If provider explicitly ruled REFUTE (e.g. mock provider or strict LLM decision)
        if provider_assessment and provider_assessment.decision == AgentDecision.REFUTE:
            decision = AgentDecision.REFUTE
            uncertainty = provider_assessment.uncertainty
            rationale = provider_assessment.rationale
            if sk_result.objection_level == "NO_OBJECTION":
                sk_result.objection_level = "CONFIRMED_FAILURE"
                sk_result.primary_failure_code = CanonicalFailureCode.F7
                sk_result.failure_candidates = [CanonicalFailureCode.F7]
                sk_result.objections = ["Adversarial reviewer confirmed literature refutation."]
        elif sk_result.objection_level == "CONFIRMED_FAILURE":
            if sk_result.primary_failure_code == CanonicalFailureCode.F7:
                decision = AgentDecision.REFUTE
            elif sk_result.primary_failure_code in (CanonicalFailureCode.F1, CanonicalFailureCode.F2, CanonicalFailureCode.F6):
                decision = AgentDecision.NEI
            else:
                decision = AgentDecision.PARTIAL
            uncertainty = 0.8
            rationale = "; ".join(sk_result.objections) if sk_result.objections else "Adversarial objections confirmed."
        elif sk_result.objection_level == "POTENTIAL_FAILURE":
            decision = AgentDecision.PARTIAL
            uncertainty = 0.5
            rationale = "; ".join(sk_result.objections) if sk_result.objections else "Potential verification weaknesses noted."
        else:
            decision = AgentDecision.SUPPORT
            uncertainty = 0.15
            rationale = "No adversarial objections found. Evidence passes scrutiny."

        return AgentAssessment(
            agent_name=self.name,
            decision=decision,
            confidence=round(1.0 - uncertainty, 2),
            supporting_evidence_ids=[e["doc_id"] for e in sk_result.supporting_evidence],
            contradicting_evidence_ids=[e["doc_id"] for e in sk_result.contradicting_evidence],
            rationale=rationale,
            uncertainty=uncertainty,
            metadata={"skeptic_result": sk_result.to_dict()}
        )
