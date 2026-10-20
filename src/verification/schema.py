"""Data schema definitions for Phase 6 multi-agent verification pipeline."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json


class AgentDecision(str, Enum):
    """Standardized verification decision options across agents."""
    SUPPORT = "SUPPORT"
    PARTIAL = "PARTIAL"
    REFUTE = "REFUTE"
    NEI = "NEI"  # Not Enough Information / Insufficient Evidence
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, val: Any) -> "AgentDecision":
        if isinstance(val, cls):
            return val
        val_str = val.value if hasattr(val, "value") else str(val)
        val_upper = val_str.strip().upper().replace(" ", "_")
        if "." in val_upper:
            val_upper = val_upper.split(".")[-1]
        if val_upper in ("SUPPORT", "SUPPORTED", "SUPPORTS"):
            return cls.SUPPORT
        elif val_upper in ("PARTIAL", "PARTIALLY_SUPPORTED", "PARTIALLY_SUPPORT"):
            return cls.PARTIAL
        elif val_upper in ("REFUTE", "REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTED"):
            return cls.REFUTE
        elif val_upper in ("NEI", "NEUTRAL", "INSUFFICIENT", "NOT_ENOUGH_INFO", "NOT ENOUGH INFORMATION"):
            return cls.NEI
        return cls.UNKNOWN


@dataclass
class VerificationRequest:
    """Input payload for verifying a scientific claim."""
    claim_text: str
    claim_id: str = "claim_000"
    dataset: str = "custom"
    split: str = "test"
    ground_truth_label: Optional[str] = None
    ground_truth_doc_ids: List[str] = field(default_factory=list)
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedEvidence:
    """Structure representing a single piece of retrieved scientific evidence."""
    doc_id: str
    text: str
    score: float = 0.0
    rank: int = 0
    source: str = "retriever"
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentAssessment:
    """Structured assessment produced by a single verification agent."""
    agent_name: str
    decision: AgentDecision
    confidence: float  # 0.0 to 1.0
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    rationale: str = ""
    uncertainty: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.decision, AgentDecision):
            self.decision = AgentDecision.from_str(self.decision)
        # Ensure confidence and uncertainty are within [0.0, 1.0]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class DebateResult:
    """Structured resolution outcome from multi-agent debate."""
    rounds: int
    initial_assessments: List[AgentAssessment]
    revised_assessments: List[AgentAssessment]
    consensus_reached: bool
    final_decision: AgentDecision
    resolution_rationale: str
    disagreement_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds": self.rounds,
            "initial_assessments": [a.to_dict() for a in self.initial_assessments],
            "revised_assessments": [a.to_dict() for a in self.revised_assessments],
            "consensus_reached": self.consensus_reached,
            "final_decision": self.final_decision.value,
            "resolution_rationale": self.resolution_rationale,
            "disagreement_reasons": self.disagreement_reasons
        }


@dataclass
class ExecutionMetrics:
    """Cost and runtime execution tracking metadata."""
    total_execution_time_sec: float = 0.0
    agent_calls_count: int = 0
    reasoning_rounds: int = 1
    retrieval_calls_count: int = 1
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """End-to-end outcome object for a scientific claim verification attempt."""
    claim_id: str
    claim_text: str
    final_label: str
    confidence: float
    evidence: List[RetrievedEvidence] = field(default_factory=list)
    agent_assessments: List[AgentAssessment] = field(default_factory=list)
    disagreement_status: bool = False
    resolution_method: str = "consensus"
    debate_result: Optional[DebateResult] = None
    failure_signature: Optional[Dict[str, Any]] = None
    run_id: str = "run_default"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "final_label": self.final_label,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "agent_assessments": [a.to_dict() for a in self.agent_assessments],
            "disagreement_status": self.disagreement_status,
            "resolution_method": self.resolution_method,
            "debate_result": self.debate_result.to_dict() if self.debate_result else None,
            "failure_signature": self.failure_signature,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "execution_metrics": self.execution_metrics.to_dict(),
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
