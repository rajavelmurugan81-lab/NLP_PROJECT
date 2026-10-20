"""Schema definitions for Research Question to Claim Generation pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid


@dataclass
class GeneratedClaim:
    """Atomic scientific claim generated from a research question."""
    claim_id: str
    question_id: str
    claim_text: str
    claim_type: str = "empirical_assertion"  # empirical_assertion, causal_hypothesis, statistical_finding
    specificity_score: float = 1.0
    is_valid: bool = True
    status: str = "GENERATED"  # GENERATED, VALIDATED, RETRIEVED, VERIFIED, REJECTED
    source_evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedClaim":
        return cls(
            claim_id=data.get("claim_id", f"claim_{uuid.uuid4().hex[:8]}"),
            question_id=data.get("question_id", "q_default"),
            claim_text=data.get("claim_text", ""),
            claim_type=data.get("claim_type", "empirical_assertion"),
            specificity_score=float(data.get("specificity_score", 1.0)),
            is_valid=bool(data.get("is_valid", True)),
            status=data.get("status", "GENERATED"),
            source_evidence=data.get("source_evidence", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ResearchQuestion:
    """User research question container."""
    question_id: str
    research_question: str
    domain: str = "general_science"
    sub_questions: List[str] = field(default_factory=list)
    generated_claims: List[GeneratedClaim] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["generated_claims"] = [c.to_dict() for c in self.generated_claims]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchQuestion":
        raw_claims = data.get("generated_claims", [])
        claims = [
            c if isinstance(c, GeneratedClaim) else GeneratedClaim.from_dict(c)
            for c in raw_claims
        ]
        return cls(
            question_id=data.get("question_id", f"q_{uuid.uuid4().hex[:8]}"),
            research_question=data.get("research_question", ""),
            domain=data.get("domain", "general_science"),
            sub_questions=data.get("sub_questions", []),
            generated_claims=claims,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {})
        )
