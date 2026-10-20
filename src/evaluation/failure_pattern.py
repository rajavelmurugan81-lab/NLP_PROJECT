"""Failure Pattern Schema and Data Models for CiteGuard.

Distinguishes specific Failure Signatures (individual observed failure instances)
from generalized reusable Failure Patterns mined from benchmark and runtime ledgers.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


@dataclass
class FailurePattern:
    """A generalized, reusable failure pattern derived from one or more failure signatures."""
    pattern_id: str
    failure_type: str
    failure_signature: str
    pattern_name: str
    severity: str
    frequency: int
    trigger_conditions: Dict[str, Any]
    observed_failure: str
    incorrect_behavior: str
    correct_behavior: str
    prevention_rule: str
    adaptive_actions: List[str] = field(default_factory=list)
    canonical_failure_code: str = ""
    canonical_failure_name: str = ""
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["canonical_category"] = f"{self.canonical_failure_code}: {self.canonical_failure_name}" if self.canonical_failure_code else self.failure_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailurePattern":
        return cls(
            pattern_id=str(data.get("pattern_id", "")),
            failure_type=str(data.get("failure_type", "UNKNOWN")),
            failure_signature=str(data.get("failure_signature", "NONE")),
            pattern_name=str(data.get("pattern_name", "UNKNOWN_PATTERN")),
            severity=str(data.get("severity", "MEDIUM")),
            frequency=int(data.get("frequency", 1)),
            trigger_conditions=dict(data.get("trigger_conditions", {})),
            observed_failure=str(data.get("observed_failure", "")),
            incorrect_behavior=str(data.get("incorrect_behavior", "")),
            correct_behavior=str(data.get("correct_behavior", "")),
            prevention_rule=str(data.get("prevention_rule", "")),
            adaptive_actions=list(data.get("adaptive_actions", [])),
            canonical_failure_code=str(data.get("canonical_failure_code", "")),
            canonical_failure_name=str(data.get("canonical_failure_name", "")),
            first_seen=str(data.get("first_seen", datetime.now(timezone.utc).isoformat())),
            last_seen=str(data.get("last_seen", datetime.now(timezone.utc).isoformat())),
            source_examples=list(data.get("source_examples", []))
        )


@dataclass
class PatternMatchResult:
    """Encapsulates the matching result between a claim/evidence situation and learned failure patterns."""
    matched: bool
    matched_pattern_ids: List[str] = field(default_factory=list)
    pattern_match_score: float = 0.0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, VERY_HIGH
    primary_pattern: Optional[FailurePattern] = None
    canonical_failure_code: str = "NONE"
    canonical_failure_name: str = "Clean Verification Contract"
    adaptive_actions: List[str] = field(default_factory=list)
    prevention_triggered: bool = False
    claim_recheck_status: str = "ACCEPT"  # ACCEPT, REVISE, REJECT
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def adaptive_action(self) -> str:
        return self.adaptive_actions[0] if self.adaptive_actions else "NONE"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["adaptive_action"] = self.adaptive_action
        if self.primary_pattern:
            d["primary_pattern"] = self.primary_pattern.to_dict()
        return d
