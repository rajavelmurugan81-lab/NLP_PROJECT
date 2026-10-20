from typing import Optional, Dict, Any
from src.verification.base_agent import BaseAgent
from src.verification.schema import AgentAssessment, AgentDecision
from src.verification.providers.base_provider import LLMProvider
from src.verification.prompts import VERIFICATION_AGENT_SYSTEM_PROMPT


class VerificationAgent(BaseAgent):
    """Lead verification agent synthesizing evidence, scope, quality, and skeptic objections into a preliminary verdict."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        super().__init__(
            name="VerificationAgent",
            provider=provider,
            system_prompt=VERIFICATION_AGENT_SYSTEM_PROMPT
        )

    def analyze(self, context: Dict[str, Any]) -> AgentAssessment:
        assessment = super().analyze(context)
        sk_res = context.get("skeptic_result")
        if sk_res:
            obj_lvl = getattr(sk_res, "objection_level", None) or (sk_res.get("objection_level") if isinstance(sk_res, dict) else None)
            p_code = getattr(sk_res, "primary_failure_code", None) or (sk_res.get("primary_failure_code") if isinstance(sk_res, dict) else None)
            p_code_str = p_code.value if hasattr(p_code, "value") else str(p_code or "")

            if obj_lvl == "CONFIRMED_FAILURE":
                if p_code_str == "F7":
                    assessment.decision = AgentDecision.REFUTE
                    assessment.rationale += " [Skeptic Revision: Confirmed contradiction F7 detected in literature.]"
                elif p_code_str in ("F1", "F2", "F6"):
                    assessment.decision = AgentDecision.NEI
                    assessment.rationale += f" [Skeptic Revision: Confirmed evidentiary/context gap {p_code_str}.]"
                elif p_code_str in ("F3", "F4", "F5", "F8"):
                    assessment.decision = AgentDecision.PARTIAL
                    assessment.rationale += f" [Skeptic Revision: Confirmed scope/reasoning limitation {p_code_str}.]"
                assessment.confidence = min(assessment.confidence, 0.60)
        return assessment
