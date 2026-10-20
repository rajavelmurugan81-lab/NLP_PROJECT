"""Unified Data Schema definition for Scientific Claim Verification datasets."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List


@dataclass
class UnifiedClaimRecord:
    """Unified record representation across all scientific claim datasets."""
    claim_id: str
    claim: str
    label: str  # SUPPORT, REFUTE, NEUTRAL, PARTIALLY_SUPPORTED, etc.
    source_dataset: str  # scifact, covert, sciclaimhunt
    split: str = "train"  # train, dev, test, val
    evidence_id: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedClaimRecord":
        """Construct record from Python dictionary."""
        return cls(
            claim_id=str(data["claim_id"]),
            claim=str(data["claim"]),
            label=str(data.get("label", "NEUTRAL")).upper(),
            source_dataset=str(data.get("source_dataset", "unknown")),
            split=str(data.get("split", "train")),
            evidence_id=str(data["evidence_id"]) if data.get("evidence_id") is not None else None,
            evidence_text=str(data["evidence_text"]) if data.get("evidence_text") is not None else None,
            evidence_title=str(data["evidence_title"]) if data.get("evidence_title") is not None else None,
            metadata=dict(data.get("metadata", {}))
        )


@dataclass
class CorpusDocumentRecord:
    """Unified document representation for corpus abstracts or full texts."""
    doc_id: str
    title: str
    abstract: List[str]  # List of sentences or paragraphs
    source_dataset: str
    structured: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusDocumentRecord":
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data.get("title", "")),
            abstract=list(data.get("abstract", [])),
            source_dataset=str(data.get("source_dataset", "unknown")),
            structured=bool(data.get("structured", False)),
            metadata=dict(data.get("metadata", {}))
        )
