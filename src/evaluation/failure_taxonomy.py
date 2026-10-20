"""Controlled Failure Taxonomy definitions for Scientific Claim Verification and Hallucination Analysis.

Authoritative source of truth for CiteGuard's Canonical F1–F8 Failure Taxonomy:
- F1: Unsupported Claim
- F2: Citation Mismatch
- F3: Overgeneralization
- F4: Causal Leap
- F5: Outdated Evidence
- F6: Population/Context Mismatch
- F7: Contradictory Evidence
- F8: Evidence Misinterpretation
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


class CanonicalFailureCode(str, Enum):
    """Authoritative F1–F8 Canonical Failure Codes for CiteGuard."""
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    F6 = "F6"
    F7 = "F7"
    F8 = "F8"


@dataclass(frozen=True)
class CanonicalFailureDefinition:
    """Rich definition and diagnostic criteria for a canonical failure category."""
    code: CanonicalFailureCode
    name: str
    description: str
    detection_criteria: str
    default_severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    adaptive_actions: List[str] = field(default_factory=list)
    legacy_aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "name": self.name,
            "description": self.description,
            "detection_criteria": self.detection_criteria,
            "default_severity": self.default_severity,
            "adaptive_actions": list(self.adaptive_actions),
            "legacy_aliases": list(self.legacy_aliases)
        }


# Canonical F1–F8 Registry
CANONICAL_FAILURE_REGISTRY: Dict[CanonicalFailureCode, CanonicalFailureDefinition] = {
    CanonicalFailureCode.F1: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F1,
        name="Unsupported Claim",
        description="Asserting or confirming claim support without qualifying empirical/textual evidence in literature.",
        detection_criteria="Zero evidence retrieved, empty passage pool, or highest evidence score < 0.70 when verdict is SUPPORTED.",
        default_severity="HIGH",
        adaptive_actions=[
            "FORCE_INSUFFICIENT_EVIDENCE",
            "REQUIRE_MULTI_PASSAGE_AGREEMENT",
            "CLAIM_RECHECK_SUFFICIENCY"
        ],
        legacy_aliases=[
            "NO_EVIDENCE",
            "INSUFFICIENT_EVIDENCE",
            "RETRIEVAL_FAILURE",
            "UNGROUNDED_SUPPORT"
        ]
    ),
    CanonicalFailureCode.F2: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F2,
        name="Citation Mismatch",
        description="Attribution of findings to a cited source, paper ID, or DOI that does not substantiate the claim, or fabricated citation metadata.",
        detection_criteria="DOI absent from registry, paper metadata missing, or cited source does not contain or mention the claim topic.",
        default_severity="CRITICAL",
        adaptive_actions=[
            "STRICT_REGISTRY_LOOKUP",
            "VERIFY_EXACT_DOI_SOURCE",
            "REJECT_UNREGISTERED_DOIS"
        ],
        legacy_aliases=[
            "CITATION_MISMATCH",
            "CITATION_FABRICATION",
            "METADATA_MISMATCH",
            "UNVERIFIED_BIBLIOGRAPHIC_METADATA"
        ]
    ),
    CanonicalFailureCode.F3: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F3,
        name="Overgeneralization",
        description="Extrapolating a narrow, qualified finding to universal certainty, or expressing high confidence on sparse/preliminary evidence.",
        detection_criteria="Confidence >= 0.85 with evidence count < 2 or relevance < 0.60; unqualified universal quantifiers ('always', 'completely').",
        default_severity="MEDIUM",
        adaptive_actions=[
            "DYNAMIC_CONFIDENCE_RECALIBRATION",
            "CAP_UNGROUNDED_CONFIDENCE",
            "DOWNGRADE_TO_PARTIALLY_SUPPORTED"
        ],
        legacy_aliases=[
            "OVERGENERALIZATION",
            "UNGROUNDED_CONFIDENCE",
            "HIGH_CONFIDENCE_ON_SPARSE_EVIDENCE"
        ]
    ),
    CanonicalFailureCode.F4: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F4,
        name="Causal Leap",
        description="Inferring direct causal mechanisms ('X causes Y', 'X directly triggers Y') from purely correlational or observational associations.",
        detection_criteria="Causal verbs asserted in claim when evidence text indicates mere association, correlation, or observational cohort data.",
        default_severity="HIGH",
        adaptive_actions=[
            "FLAG_CAUSAL_LEAP",
            "MANDATE_CORRELATIONAL_QUALIFIER",
            "REVISE_OR_REJECT_CLAIM"
        ],
        legacy_aliases=[
            "CAUSAL_LEAP",
            "CAUSAL_EXTRAPOLATION",
            "CORRELATION_AS_CAUSATION"
        ]
    ),
    CanonicalFailureCode.F5: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F5,
        name="Outdated Evidence",
        description="Relying on superseded, retracted, or chronologically outdated findings where newer literature contradicts or modifies the premise.",
        detection_criteria="Evidence publication year is significantly outdated relative to target year, or paper has been flagged as retracted.",
        default_severity="MEDIUM",
        adaptive_actions=[
            "CHECK_TEMPORAL_RECENCY",
            "EXCLUDE_RETRACTED_PAPERS",
            "FILTER_OUTDATED_STUDIES"
        ],
        legacy_aliases=[
            "OUTDATED_EVIDENCE",
            "OUTDATED_SOURCE",
            "TEMPORAL_MISMATCH"
        ]
    ),
    CanonicalFailureCode.F6: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F6,
        name="Population/Context Mismatch",
        description="Applying findings established in one specific population, organism, species, or scientific discipline to an incompatible target context.",
        detection_criteria="Cross-domain topic conflict (e.g., smart grid papers for quantum FL) or species/experimental model leap (in-vitro to human clinical).",
        default_severity="HIGH",
        adaptive_actions=[
            "REQUIRE_STRICT_DOMAIN_VALIDATION",
            "EXCLUDE_OFF_DOMAIN_PASSAGES",
            "CLAIM_RECHECK_DOMAIN_SCOPE"
        ],
        legacy_aliases=[
            "DOMAIN_MISMATCH",
            "WRONG_EVIDENCE",
            "SCOPE_MISMATCH",
            "SOURCE_QUALITY_FAILURE"
        ]
    ),
    CanonicalFailureCode.F7: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F7,
        name="Contradictory Evidence",
        description="Sustaining a SUPPORTED verdict despite explicit refutations, counter-evidence, or directionality inversions in literature.",
        detection_criteria="Contradicting passages count > 0 with SUPPORT verdict; directional flips ('increases' vs 'decreases').",
        default_severity="CRITICAL",
        adaptive_actions=[
            "ESCALATE_TO_MULTI_AGENT_DEBATE",
            "MANDATE_CONTRADICTION_DEBATE",
            "FORCE_CONTRADICTED_OR_PARTIAL"
        ],
        legacy_aliases=[
            "CONTRADICTORY_EVIDENCE",
            "DIRECTIONALITY_INVERSION"
        ]
    ),
    CanonicalFailureCode.F8: CanonicalFailureDefinition(
        code=CanonicalFailureCode.F8,
        name="Evidence Misinterpretation",
        description="Distorting or misreading statistical findings, p-values, directionality, or semantic relationships in evidence passages.",
        detection_criteria="Inverted statistical inferences, numerical distortions, or reasoning classification errors.",
        default_severity="HIGH",
        adaptive_actions=[
            "ACTIVATE_ADVERSARIAL_REFUTER",
            "VERIFY_EXACT_INTERVALS",
            "REASONING_AUDIT"
        ],
        legacy_aliases=[
            "REASONING_FAILURE",
            "LABEL_FAILURE",
            "NUMERICAL_DISTORTION",
            "CLAIM_MALFORMATION"
        ]
    ),
}


class FailureType(str, Enum):
    """FailureType enum preserving backward compatibility with legacy tests while embedding canonical F1–F8."""

    # Canonical F1–F8 types
    F1_UNSUPPORTED_CLAIM = "F1_UNSUPPORTED_CLAIM"
    F2_CITATION_MISMATCH = "F2_CITATION_MISMATCH"
    F3_OVERGENERALIZATION = "F3_OVERGENERALIZATION"
    F4_CAUSAL_LEAP = "F4_CAUSAL_LEAP"
    F5_OUTDATED_EVIDENCE = "F5_OUTDATED_EVIDENCE"
    F6_CONTEXT_MISMATCH = "F6_CONTEXT_MISMATCH"
    F7_CONTRADICTORY_EVIDENCE = "F7_CONTRADICTORY_EVIDENCE"
    F8_EVIDENCE_MISINTERPRETATION = "F8_EVIDENCE_MISINTERPRETATION"

    # Canonical code shorthands
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    F6 = "F6"
    F7 = "F7"
    F8 = "F8"

    # Legacy Backward-Compatibility Types (Exact names expected by existing test suite)
    NO_EVIDENCE = "NO_EVIDENCE"
    WRONG_EVIDENCE = "WRONG_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    REASONING_FAILURE = "REASONING_FAILURE"
    LABEL_FAILURE = "LABEL_FAILURE"
    SOURCE_QUALITY_FAILURE = "SOURCE_QUALITY_FAILURE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

    # Auxiliary diagnostic signals
    DOMAIN_MISMATCH = "DOMAIN_MISMATCH"
    UNGROUNDED_SUPPORT = "UNGROUNDED_SUPPORT"
    UNGROUNDED_CONFIDENCE = "UNGROUNDED_CONFIDENCE"
    CLEAN = "CLEAN"
    NONE = "NONE"

    def to_canonical_code(self) -> Optional[CanonicalFailureCode]:
        """Map any failure type or legacy member to its CanonicalFailureCode."""
        val = self.value.upper()
        if val in ("F1", "F1_UNSUPPORTED_CLAIM", "NO_EVIDENCE", "INSUFFICIENT_EVIDENCE", "RETRIEVAL_FAILURE", "UNGROUNDED_SUPPORT"):
            return CanonicalFailureCode.F1
        if val in ("F2", "F2_CITATION_MISMATCH", "CITATION_FABRICATION", "METADATA_MISMATCH"):
            return CanonicalFailureCode.F2
        if val in ("F3", "F3_OVERGENERALIZATION", "UNGROUNDED_CONFIDENCE", "PARTIAL_EVIDENCE"):
            return CanonicalFailureCode.F3
        if val in ("F4", "F4_CAUSAL_LEAP", "CAUSAL_EXTRAPOLATION"):
            return CanonicalFailureCode.F4
        if val in ("F5", "F5_OUTDATED_EVIDENCE", "OUTDATED_SOURCE", "TEMPORAL_MISMATCH"):
            return CanonicalFailureCode.F5
        if val in ("F6", "F6_CONTEXT_MISMATCH", "DOMAIN_MISMATCH", "WRONG_EVIDENCE", "SCOPE_MISMATCH", "SOURCE_QUALITY_FAILURE"):
            return CanonicalFailureCode.F6
        if val in ("F7", "F7_CONTRADICTORY_EVIDENCE", "CONTRADICTORY_EVIDENCE", "DIRECTIONALITY_INVERSION"):
            return CanonicalFailureCode.F7
        if val in ("F8", "F8_EVIDENCE_MISINTERPRETATION", "REASONING_FAILURE", "LABEL_FAILURE", "NUMERICAL_DISTORTION", "CLAIM_MALFORMATION"):
            return CanonicalFailureCode.F8
        return None

    def to_canonical_definition(self) -> Optional[CanonicalFailureDefinition]:
        """Get the full CanonicalFailureDefinition for this failure type."""
        code = self.to_canonical_code()
        return CANONICAL_FAILURE_REGISTRY.get(code) if code else None


# Backward-compatible description lookup dictionary
FAILURE_TAXONOMY_DESCRIPTIONS: Dict[FailureType, str] = {
    FailureType.NO_EVIDENCE: "No sufficiently relevant evidence was retrieved from the corpus.",
    FailureType.WRONG_EVIDENCE: "Retrieved evidence does not support or address the claim decision.",
    FailureType.INSUFFICIENT_EVIDENCE: "Retrieved evidence is incomplete or insufficient to establish the claim.",
    FailureType.CONTRADICTORY_EVIDENCE: "Retrieved evidence conflicts with the predicted conclusion.",
    FailureType.PARTIAL_EVIDENCE: "Evidence supports only part of a complex claim.",
    FailureType.RETRIEVAL_FAILURE: "Ground-truth evidence exists in corpus but was not retrieved in top-K.",
    FailureType.REASONING_FAILURE: "Ground-truth evidence was retrieved, but model reached incorrect verdict.",
    FailureType.LABEL_FAILURE: "Evidence reasoning is adequate, but label mapping failed.",
    FailureType.SOURCE_QUALITY_FAILURE: "Retrieved source document is contextually inappropriate or outdated.",
    FailureType.UNKNOWN_FAILURE: "Failure cannot be confidently assigned to a specific category.",
    # Canonical additions
    FailureType.F1: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F1].description,
    FailureType.F2: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F2].description,
    FailureType.F3: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F3].description,
    FailureType.F4: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F4].description,
    FailureType.F5: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F5].description,
    FailureType.F6: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F6].description,
    FailureType.F7: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F7].description,
    FailureType.F8: CANONICAL_FAILURE_REGISTRY[CanonicalFailureCode.F8].description,
}


def get_failure_type_description(failure_type: FailureType) -> str:
    """Retrieve human-readable description for a given failure type."""
    return FAILURE_TAXONOMY_DESCRIPTIONS.get(failure_type, "Unknown failure type")


def map_legacy_to_canonical(signal: str) -> Optional[CanonicalFailureDefinition]:
    """Map any string signal or legacy category name to its canonical F1–F8 definition."""
    if not signal:
        return None
    s = signal.strip().upper()
    try:
        ft = FailureType(s)
        return ft.to_canonical_definition()
    except (ValueError, KeyError):
        pass

    # Direct code check
    for code, defn in CANONICAL_FAILURE_REGISTRY.items():
        if s == code.value or s == defn.name.upper():
            return defn
        for alias in defn.legacy_aliases:
            if s == alias.upper():
                return defn
    return None


# Alias for canonical failure taxonomy
CanonicalFailureType = CanonicalFailureCode
