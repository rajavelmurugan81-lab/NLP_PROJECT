"""Multi-agent debate resolution engine for CiteGuard."""

import logging
from typing import List, Dict, Any, Optional
import json

from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence, DebateResult
from src.verification.providers.base_provider import LLMProvider
from src.verification.providers.factory import get_provider
from src.verification.prompts import DEBATE_RESOLVER_SYSTEM_PROMPT, format_agent_prompt
from src.verification.policy import VerificationPolicy

logger = logging.getLogger(__name__)


class DebateResolver:
    """Orchestrates structured comparative debate to resolve multi-agent disagreement."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_provider("mock")

    def resolve(
        self,
        claim_text: str,
        evidence: List[RetrievedEvidence],
        assessments: List[AgentAssessment],
        policy: Optional[VerificationPolicy] = None
    ) -> DebateResult:
        """Execute debate resolution round over conflicting agent assessments."""
        policy = policy or VerificationPolicy()

        prompt = self._build_debate_prompt(claim_text, evidence, assessments)

        try:
            res_dict = self.provider.generate_structured(prompt, system_prompt=DEBATE_RESOLVER_SYSTEM_PROMPT)
            
            final_decision_raw = res_dict.get("final_decision", "NEI")
            final_decision = AgentDecision.from_str(final_decision_raw)
            consensus_reached = bool(res_dict.get("consensus_reached", False))
            rationale = str(res_dict.get("resolution_rationale", "Debate synthesis completed."))
            disagreements = list(res_dict.get("disagreement_reasons", []))

            # Revised assessments reflect consensus alignment
            revised_assessments = []
            for a in assessments:
                revised_assessments.append(
                    AgentAssessment(
                        agent_name=a.agent_name,
                        decision=final_decision if consensus_reached else a.decision,
                        confidence=min(1.0, a.confidence + 0.1) if consensus_reached else a.confidence,
                        supporting_evidence_ids=a.supporting_evidence_ids,
                        contradicting_evidence_ids=a.contradicting_evidence_ids,
                        rationale=a.rationale,
                        uncertainty=a.uncertainty,
                        metadata={"revised_in_debate": True}
                    )
                )

            return DebateResult(
                rounds=1,
                initial_assessments=assessments,
                revised_assessments=revised_assessments,
                consensus_reached=consensus_reached,
                final_decision=final_decision,
                resolution_rationale=rationale,
                disagreement_reasons=disagreements
            )

        except Exception as e:
            logger.error(f"Error during debate resolution: {str(e)}")
            return DebateResult(
                rounds=1,
                initial_assessments=assessments,
                revised_assessments=assessments,
                consensus_reached=False,
                final_decision=AgentDecision.NEI,
                resolution_rationale=f"Debate resolution failed due to provider error: {str(e)}",
                disagreement_reasons=[str(e)]
            )

    def _build_debate_prompt(
        self,
        claim_text: str,
        evidence: List[RetrievedEvidence],
        assessments: List[AgentAssessment]
    ) -> str:
        evidence_dicts = [e.to_dict() for e in evidence]
        base_prompt = format_agent_prompt(claim_text, evidence_dicts)

        lines = [
            base_prompt,
            "\n=== CONFLICTING AGENT ASSESSMENTS TO RESOLVE ==="
        ]

        for idx, a in enumerate(assessments, 1):
            lines.append(f"Agent [{idx}]: {a.agent_name}")
            lines.append(f"  Decision: {a.decision.value}")
            lines.append(f"  Confidence: {a.confidence:.2f}")
            lines.append(f"  Supporting Doc IDs: {a.supporting_evidence_ids}")
            lines.append(f"  Contradicting Doc IDs: {a.contradicting_evidence_ids}")
            lines.append(f"  Rationale: {a.rationale}\n")

        lines.append("Compare agent rationales against the evidence passages. Resolve the verdict strictly based on evidence quality and grounding.")
        return "\n".join(lines)
