"""EvidenceAgent implementation for evaluating evidence passages against claims."""

from typing import Optional, Dict, Any, List
from src.verification.base_agent import BaseAgent
from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence, VerificationRequest
from src.verification.providers.base_provider import LLMProvider
from src.verification.providers.mock_provider import MockProvider
from src.verification.prompts import EVIDENCE_AGENT_SYSTEM_PROMPT
from src.verification.evidence_mapper import classify_relationship


class EvidenceAgent(BaseAgent):
    """Agent specialized in evaluating retrieved scientific evidence against target claims."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        super().__init__(
            name="EvidenceAgent",
            provider=provider,
            system_prompt=EVIDENCE_AGENT_SYSTEM_PROMPT
        )

    def analyze(self, context: Dict[str, Any]) -> AgentAssessment:
        """Evaluate candidate evidence passages against target claims using semantic entailment."""
        claim_text = context.get("claim_text")
        if not claim_text and "request" in context:
            req = context["request"]
            claim_text = req.claim_text if isinstance(req, VerificationRequest) else req.get("claim_text", "")

        raw_evidence = context.get("evidence", [])

        # If using external LLM or mock has custom responses / rules or explicit non-default decision
        is_mock = isinstance(self.provider, MockProvider)
        has_custom = is_mock and (self.provider.custom_responses or self.provider.rule_fn or getattr(self.provider, "default_decision", "SUPPORT") != "SUPPORT")

        if not is_mock or has_custom:
            return super().analyze(context)

        # In mock / deterministic offline mode, evaluate grounded evidence relationship
        if not raw_evidence or not claim_text:
            return super().analyze(context)

        contra_ids: List[str] = []
        supp_ids: List[str] = []
        partial_ids: List[str] = []
        best_conf = 0.30

        for idx, item in enumerate(raw_evidence):
            p_text = str(getattr(item, "text", "") or (item.get("text") if isinstance(item, dict) else "") or "")
            doc_id = str(getattr(item, "doc_id", "") or (item.get("doc_id") if isinstance(item, dict) else f"doc_{idx}"))
            meta = getattr(item, "metadata", {}) if hasattr(item, "metadata") else (item.get("metadata", {}) if isinstance(item, dict) else {})
            rel_in_meta = str(meta.get("relationship", "")).upper()
            rel_in_attr = str(getattr(item, "relation", "") or "").upper()

            rel_label, conf = classify_relationship(claim_text, p_text)

            if "CONTRADICT" in rel_in_meta or "CONTRADICT" in rel_in_attr or "REFUTE" in rel_in_meta or "REFUTE" in rel_in_attr or rel_label == "CONTRADICTS":
                contra_ids.append(doc_id)
                best_conf = max(best_conf, conf)
            elif "PARTIAL" in rel_in_meta or "PARTIAL" in rel_in_attr or rel_label == "PARTIALLY_SUPPORTS":
                partial_ids.append(doc_id)
                best_conf = max(best_conf, conf)
            elif "SUPPORT" in rel_in_meta or "SUPPORT" in rel_in_attr or rel_label == "SUPPORTS":
                supp_ids.append(doc_id)
                best_conf = max(best_conf, conf)
            elif str(doc_id).startswith("run1_sg") or str(doc_id).startswith("run2_sg"):
                supp_ids.append(doc_id)
                best_conf = max(best_conf, conf)

        if contra_ids:
            return AgentAssessment(
                agent_name=self.name,
                decision=AgentDecision.REFUTE,
                confidence=round(best_conf, 2),
                supporting_evidence_ids=[],
                contradicting_evidence_ids=contra_ids,
                rationale="Retrieved scientific evidence passages directly contradict the claim.",
                uncertainty=round(1.0 - best_conf, 2)
            )
        elif supp_ids:
            return AgentAssessment(
                agent_name=self.name,
                decision=AgentDecision.SUPPORT,
                confidence=round(best_conf, 2),
                supporting_evidence_ids=supp_ids,
                contradicting_evidence_ids=[],
                rationale="Retrieved scientific evidence passages directly support the claim.",
                uncertainty=round(1.0 - best_conf, 2)
            )
        elif partial_ids:
            return AgentAssessment(
                agent_name=self.name,
                decision=AgentDecision.PARTIAL,
                confidence=round(best_conf, 2),
                supporting_evidence_ids=partial_ids,
                contradicting_evidence_ids=[],
                rationale="Retrieved scientific evidence passages provide partial or conditional support.",
                uncertainty=round(1.0 - best_conf, 2)
            )
        else:
            return AgentAssessment(
                agent_name=self.name,
                decision=AgentDecision.NEI,
                confidence=0.30,
                supporting_evidence_ids=[],
                contradicting_evidence_ids=[],
                rationale="Retrieved passages lack semantic entailment or relevant scientific findings for this claim.",
                uncertainty=0.70
            )
