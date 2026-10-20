"""Structured Failure Signature schema for scientific claim verification errors."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import hashlib
from src.evaluation.failure_taxonomy import FailureType


@dataclass
class FailureSignature:
    """Structured record encapsulating a single detected claim verification failure."""
    
    claim_id: str
    claim_text: str
    dataset: str
    predicted_label: str
    ground_truth_label: str
    failure_type: FailureType
    split: str = "train"
    retrieved_document_ids: List[str] = field(default_factory=list)
    retrieved_ranks: List[int] = field(default_factory=list)
    retrieval_mode: str = "hybrid"
    evidence_available: bool = True
    evidence_retrieved: bool = False
    failure_subtype: Optional[str] = None
    confidence: float = 0.0
    explanation: str = ""
    signature_id: str = ""
    run_id: str = "run_default"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate failure_type is a valid FailureType Enum member
        if isinstance(self.failure_type, str):
            try:
                self.failure_type = FailureType(self.failure_type)
            except ValueError:
                self.failure_type = FailureType.UNKNOWN_FAILURE
        if not self.signature_id:
            h_suffix = self.compute_signature_hash()[:4].upper()
            self.signature_id = f"FS-{h_suffix}"

    def compute_signature_hash(self) -> str:
        """Compute unique deterministic hash key for duplicate ledger prevention."""
        raw_key = f"{self.run_id}_{self.dataset}_{self.claim_id}_{self.retrieval_mode}_{self.failure_type.value}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert FailureSignature to dictionary format."""
        d = asdict(self)
        d["failure_type"] = self.failure_type.value if isinstance(self.failure_type, FailureType) else str(self.failure_type)
        d["signature_hash"] = self.compute_signature_hash()
        d["signature_id"] = self.signature_id or f"FS-{self.compute_signature_hash()[:4].upper()}"
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureSignature":
        """Construct FailureSignature from dictionary data."""
        ft_raw = data.get("failure_type", "UNKNOWN_FAILURE")
        try:
            ft = FailureType(ft_raw)
        except ValueError:
            ft = FailureType.UNKNOWN_FAILURE

        return cls(
            claim_id=str(data["claim_id"]),
            claim_text=str(data["claim_text"]),
            dataset=str(data.get("dataset", "unknown")),
            split=str(data.get("split", "train")),
            predicted_label=str(data.get("predicted_label", "UNKNOWN")),
            ground_truth_label=str(data.get("ground_truth_label", "UNKNOWN")),
            failure_type=ft,
            retrieved_document_ids=list(data.get("retrieved_document_ids", [])),
            retrieved_ranks=list(data.get("retrieved_ranks", [])),
            retrieval_mode=str(data.get("retrieval_mode", "hybrid")),
            evidence_available=bool(data.get("evidence_available", True)),
            evidence_retrieved=bool(data.get("evidence_retrieved", False)),
            failure_subtype=data.get("failure_subtype"),
            confidence=float(data.get("confidence", 0.0)),
            explanation=str(data.get("explanation", "")),
            signature_id=str(data.get("signature_id", "")),
            run_id=str(data.get("run_id", "run_default")),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            metadata=dict(data.get("metadata", {}))
        )
