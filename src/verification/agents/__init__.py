"""Agent roles package for CiteGuard verification framework."""

from src.verification.agents.evidence_agent import EvidenceAgent
from src.verification.agents.claim_analysis_agent import ClaimAnalysisAgent
from src.verification.agents.source_quality_agent import SourceQualityAgent
from src.verification.agents.verification_agent import VerificationAgent
from src.verification.skeptic_agent import SkepticAgent, SkepticResult

__all__ = [
    "EvidenceAgent",
    "ClaimAnalysisAgent",
    "SourceQualityAgent",
    "VerificationAgent",
    "SkepticAgent",
    "SkepticResult",
]
