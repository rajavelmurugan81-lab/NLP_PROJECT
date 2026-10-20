"""Adaptive verification policy and routing threshold rules."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from src.verification.schema import AgentAssessment, AgentDecision, RetrievedEvidence

logger = logging.getLogger(__name__)


@dataclass
class VerificationPolicy:
    """Configurable adaptive verification routing thresholds."""

    confidence_threshold: float = 0.70
    agreement_threshold: float = 0.75
    evidence_quality_threshold: float = 0.001
    max_reasoning_rounds: int = 3
    max_retrieved_evidence: int = 5
    fallback_behavior: str = "NEI"

    def evaluate_agreement(self, assessments: List[AgentAssessment]) -> Dict[str, Any]:
        """Calculate decision agreement distribution across agents."""
        if not assessments:
            return {
                "agreement_ratio": 0.0,
                "majority_decision": AgentDecision.NEI,
                "disagreement_status": True,
                "decision_counts": {}
            }

        decision_counts: Dict[str, int] = {}
        for a in assessments:
            val = a.decision.value
            decision_counts[val] = decision_counts.get(val, 0) + 1

        total = len(assessments)
        majority_val, max_count = max(decision_counts.items(), key=lambda item: item[1])
        ratio = max_count / float(total)

        disagreement = ratio < self.agreement_threshold

        return {
            "agreement_ratio": ratio,
            "majority_decision": AgentDecision.from_str(majority_val),
            "disagreement_status": disagreement,
            "decision_counts": decision_counts
        }

    def should_trigger_debate(self, assessments: List[AgentAssessment]) -> bool:
        """Determine whether multi-agent disagreement debate should be triggered."""
        if not assessments:
            return False

        agreement_info = self.evaluate_agreement(assessments)
        if agreement_info["disagreement_status"]:
            return True

        # Check if mean confidence is below threshold
        mean_conf = sum(a.confidence for a in assessments) / float(len(assessments))
        if mean_conf < self.confidence_threshold:
            return True

        return False

    def is_evidence_sufficient(self, evidence: List[RetrievedEvidence]) -> bool:
        """Check if retrieved evidence meets minimum count and score requirements."""
        if not evidence:
            return False

        max_score = max(e.score for e in evidence) if evidence else 0.0
        return max_score >= self.evidence_quality_threshold
