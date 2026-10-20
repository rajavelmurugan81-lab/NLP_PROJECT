"""Abstract Base Agent class for CiteGuard verification framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence, VerificationRequest
from src.verification.providers.base_provider import LLMProvider
from src.verification.providers.factory import get_provider
from src.verification.prompts import format_agent_prompt

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all verification agents."""

    def __init__(
        self,
        name: str,
        provider: Optional[LLMProvider] = None,
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.provider = provider or get_provider("mock")
        self.system_prompt = system_prompt or ""

    def analyze(self, context: Dict[str, Any]) -> AgentAssessment:
        """Standard analysis entry point for verification context."""
        claim_text = context.get("claim_text")
        if not claim_text and "request" in context:
            req = context["request"]
            claim_text = req.claim_text if isinstance(req, VerificationRequest) else req.get("claim_text", "")

        raw_evidence = context.get("evidence", [])
        evidence_dicts = []
        for item in raw_evidence:
            if isinstance(item, RetrievedEvidence):
                evidence_dicts.append(item.to_dict())
            elif isinstance(item, dict):
                evidence_dicts.append(item)

        user_prompt = self._build_prompt(claim_text, evidence_dicts, context)

        try:
            res_dict = self.provider.generate_structured(user_prompt, system_prompt=self.system_prompt)
            return self._parse_assessment(res_dict)
        except Exception as e:
            logger.error(f"[{self.name}] Error during analysis: {str(e)}")
            return AgentAssessment(
                agent_name=self.name,
                decision=AgentDecision.NEI,
                confidence=0.0,
                rationale=f"Agent analysis failed due to provider error: {str(e)}",
                uncertainty=1.0,
                metadata={"error": str(e)}
            )

    def _build_prompt(self, claim_text: str, evidence_dicts: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Construct user prompt. Override in subclasses if specialized prompt format is needed."""
        return format_agent_prompt(claim_text, evidence_dicts)

    def _parse_assessment(self, res_dict: Dict[str, Any]) -> AgentAssessment:
        """Parse dictionary response into typed AgentAssessment."""
        decision_raw = res_dict.get("decision", "NEI")
        decision = AgentDecision.from_str(decision_raw)

        confidence = float(res_dict.get("confidence", 0.5))
        supporting_ids = list(res_dict.get("supporting_evidence_ids", []))
        contradicting_ids = list(res_dict.get("contradicting_evidence_ids", []))
        rationale = str(res_dict.get("rationale", "No rationale provided."))
        uncertainty = float(res_dict.get("uncertainty", 1.0 - confidence))

        return AgentAssessment(
            agent_name=self.name,
            decision=decision,
            confidence=confidence,
            supporting_evidence_ids=supporting_ids,
            contradicting_evidence_ids=contradicting_ids,
            rationale=rationale,
            uncertainty=uncertainty,
            metadata=res_dict.get("metadata", {})
        )
