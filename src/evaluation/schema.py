"""Evaluation Schema module for CiteGuard benchmark framework."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json


@dataclass
class EvaluationExample:
    """Standardized evaluation input sample."""
    example_id: str
    claim_text: str
    dataset: str = "scifact"
    split: str = "test"
    ground_truth_label: str = "NEI"
    ground_truth_doc_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    """Individual system prediction record for evaluation tracing."""
    example_id: str
    claim_text: str
    predicted_label: str
    ground_truth_label: str = "UNKNOWN"
    confidence: float = 0.0
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieved_ranks: List[int] = field(default_factory=list)
    system_name: str = "system_default"
    execution_time_sec: float = 0.0
    agent_calls: int = 0
    reasoning_rounds: int = 1
    run_id: str = "run_default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalMetrics:
    """Information retrieval metric scores."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "mrr": round(self.mrr, 4),
            "hit_at_k": {str(k): round(v, 4) for k, v in self.hit_at_k.items()},
            "ndcg_at_k": {str(k): round(v, 4) for k, v in self.ndcg_at_k.items()}
        }


@dataclass
class ClassificationMetrics:
    """Verification classification metric scores."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    class_report: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "confusion_matrix": self.confusion_matrix,
            "class_report": self.class_report
        }


@dataclass
class MultiAgentMetrics:
    """Agent orchestration performance and efficiency metrics."""
    total_examples: int = 0
    agents_invoked: int = 0
    reasoning_rounds: float = 1.0
    agreement_rate: float = 1.0
    disagreement_rate: float = 0.0
    debate_activation_rate: float = 0.0
    average_execution_time_sec: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "agents_invoked": self.agents_invoked,
            "reasoning_rounds": round(self.reasoning_rounds, 2),
            "agreement_rate": round(self.agreement_rate, 4),
            "disagreement_rate": round(self.disagreement_rate, 4),
            "debate_activation_rate": round(self.debate_activation_rate, 4),
            "average_execution_time_sec": round(self.average_execution_time_sec, 4),
            "token_usage": self.token_usage
        }


@dataclass
class RHRMetrics:
    """Repeated Hallucination Rate (RHR) benchmark metrics."""
    repeated_hallucination_rate: float = 0.0
    stateless_rhr: float = 0.0
    rhr_reduction_percent: float = 0.0
    total_failures: int = 0
    repeated_failures_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repeated_hallucination_rate": round(self.repeated_hallucination_rate, 4),
            "stateless_rhr": round(self.stateless_rhr, 4),
            "rhr_reduction_percent": round(self.rhr_reduction_percent, 2),
            "total_failures": self.total_failures,
            "repeated_failures_count": self.repeated_failures_count
        }


@dataclass
class EvaluationResult:
    """Comprehensive evaluation experiment results container."""
    run_id: str
    system_name: str
    dataset: str
    split: str
    total_examples: int
    retrieval_metrics: Optional[RetrievalMetrics] = None
    classification_metrics: Optional[ClassificationMetrics] = None
    multi_agent_metrics: Optional[MultiAgentMetrics] = None
    rhr_metrics: Optional[RHRMetrics] = None
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "system_name": self.system_name,
            "dataset": self.dataset,
            "split": self.split,
            "total_examples": self.total_examples,
            "retrieval_metrics": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            "classification_metrics": self.classification_metrics.to_dict() if self.classification_metrics else None,
            "multi_agent_metrics": self.multi_agent_metrics.to_dict() if self.multi_agent_metrics else None,
            "rhr_metrics": self.rhr_metrics.to_dict() if self.rhr_metrics else None,
            "failure_breakdown": self.failure_breakdown,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
