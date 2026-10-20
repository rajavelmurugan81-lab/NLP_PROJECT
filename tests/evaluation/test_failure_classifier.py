"""Comprehensive test suite for the canonical F1–F8 failure taxonomy and classifier.

Tests:
1. Canonical failure codes and registry integrity (F1 through F8).
2. Legacy-to-canonical mapping for backward compatibility.
3. Every individual failure category classification:
   - F1: Unsupported Claim
   - F2: Citation Mismatch
   - F3: Overgeneralization
   - F4: Causal Leap
   - F5: Outdated Evidence
   - F6: Population/Context Mismatch
   - F7: Contradictory Evidence
   - F8: Evidence Misinterpretation
4. Clean verification contract (no failure detected).
5. Integration with RuntimeAuditor, PatternMiner, and FailureMemory.
"""

from dataclasses import dataclass, field
import pytest

from src.evaluation.failure_taxonomy import (
    CanonicalFailureCode,
    CANONICAL_FAILURE_REGISTRY,
    FailureType,
    map_legacy_to_canonical,
    get_failure_type_description
)
from src.evaluation.failure_classifier import (
    FailureClassifier,
    FailureClassificationResult
)
from src.evaluation.runtime_auditor import RuntimeAuditor, RuntimeAuditRecord
from src.evaluation.pattern_miner import PatternMiner
from src.evaluation.failure_memory import FailureMemory


@dataclass
class MockPassage:
    doc_id: str = "doc_1"
    text: str = ""
    score: float = 0.0
    rank: int = 1
    relation: str = ""
    metadata: dict = field(default_factory=dict)


# ============================================================================
# 1. TAXONOMY REGISTRY INTEGRITY
# ============================================================================

def test_canonical_codes_complete():
    """All 8 canonical codes must exist in the enum."""
    expected_codes = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    enum_codes = [c.value for c in CanonicalFailureCode]
    assert enum_codes == expected_codes


def test_canonical_registry_coverage():
    """The canonical registry must contain definitions for all F1–F8 codes."""
    assert len(CANONICAL_FAILURE_REGISTRY) == 8
    for code in CanonicalFailureCode:
        defn = CANONICAL_FAILURE_REGISTRY.get(code)
        assert defn is not None
        assert defn.code == code.value
        assert len(defn.name) > 0
        assert len(defn.description) > 0
        assert len(defn.detection_criteria) > 0
        assert defn.default_severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert len(defn.adaptive_actions) > 0


def test_legacy_to_canonical_mapping():
    """Legacy enum values and strings must deterministically map to F1–F8."""
    assert map_legacy_to_canonical(FailureType.NO_EVIDENCE).code == CanonicalFailureCode.F1
    assert map_legacy_to_canonical(FailureType.RETRIEVAL_FAILURE).code == CanonicalFailureCode.F1
    assert map_legacy_to_canonical(FailureType.INSUFFICIENT_EVIDENCE).code == CanonicalFailureCode.F1
    assert map_legacy_to_canonical(FailureType.WRONG_EVIDENCE).code == CanonicalFailureCode.F6
    assert map_legacy_to_canonical(FailureType.DOMAIN_MISMATCH).code == CanonicalFailureCode.F6
    assert map_legacy_to_canonical("DOMAIN_MISMATCH").code == CanonicalFailureCode.F6
    assert map_legacy_to_canonical("CONTRADICTORY_EVIDENCE").code == CanonicalFailureCode.F7
    assert map_legacy_to_canonical("CITATION_MISMATCH").code == CanonicalFailureCode.F2
    assert map_legacy_to_canonical("OVERGENERALIZATION").code == CanonicalFailureCode.F3
    assert map_legacy_to_canonical("CAUSAL_LEAP").code == CanonicalFailureCode.F4
    assert map_legacy_to_canonical("OUTDATED_EVIDENCE").code == CanonicalFailureCode.F5
    assert map_legacy_to_canonical("CLAIM_MALFORMATION").code == CanonicalFailureCode.F8
    assert map_legacy_to_canonical(FailureType.F1).code == CanonicalFailureCode.F1
    assert map_legacy_to_canonical(FailureType.F8).code == CanonicalFailureCode.F8


# ============================================================================
# 2. INDIVIDUAL F1–F8 CLASSIFICATION TESTS
# ============================================================================

@pytest.fixture
def classifier():
    return FailureClassifier()


def test_f1_unsupported_claim(classifier):
    """F1: Claim marked SUPPORT despite having zero evidence or subthreshold scores."""
    # Zero evidence case
    res1 = classifier.classify(
        claim_text="Quantum key distribution guarantees absolute security on all channels.",
        verdict="SUPPORT",
        confidence=0.88,
        evidence=[]
    )
    assert res1.has_failure is True
    assert res1.canonical_code == CanonicalFailureCode.F1
    assert "Unsupported" in res1.canonical_name
    assert res1.severity in ("HIGH", "CRITICAL")

    # Subthreshold score case
    weak_ev = [
        MockPassage(
            doc_id="doc1",
            text="Quantum protocols have been tested under laboratory conditions.",
            score=0.45,
            rank=1
        )
    ]
    res2 = classifier.classify(
        claim_text="Quantum key distribution is practically invincible in all networks.",
        verdict="SUPPORT",
        confidence=0.85,
        evidence=weak_ev
    )
    assert res2.has_failure is True
    assert res2.canonical_code == CanonicalFailureCode.F1


def test_f2_citation_mismatch(classifier):
    """F2: Citation points to wrong paper or metadata fabrication."""
    meta_mismatch_ev = [
        MockPassage(
            doc_id="doc2",
            text="Federated learning converges in decentralized setups.",
            score=0.82,
            rank=1,
            metadata={"metadata_unverified": True, "doi": None, "relationship": "SUPPORT"}
        )
    ]
    res = classifier.classify(
        claim_text="FL achieves 95% accuracy in 2026 as shown in DOI 10.1234/fake.doi.",
        verdict="SUPPORT",
        confidence=0.90,
        evidence=meta_mismatch_ev,
        question="Find 10 papers with exact DOI and citation count",
        question_type="BIBLIOGRAPHIC_SEARCH"
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F2
    assert "Citation Mismatch" in res.canonical_name


def test_f3_overgeneralization(classifier):
    """F3: Broad universal claims supported by narrow or single evidence, or high confidence on sparse evidence."""
    ev = [
        MockPassage(
            doc_id="doc3",
            text="A single trial of 12 patients in Texas showed reduced fever.",
            score=0.72,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    res = classifier.classify(
        claim_text="The treatment is universally effective for all human populations worldwide.",
        verdict="SUPPORT",
        confidence=0.95,
        evidence=ev
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F3
    assert "Overgeneralization" in res.canonical_name


def test_f4_causal_leap(classifier):
    """F4: Asserting causation from correlation without causal evidence."""
    ev = [
        MockPassage(
            doc_id="doc4",
            text="Ice cream sales and drowning incidents both increase during summer months.",
            score=0.78,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    res = classifier.classify(
        claim_text="Eating ice cream directly causes an increase in drowning incidents.",
        verdict="SUPPORT",
        confidence=0.88,
        evidence=ev
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F4
    assert "Causal Leap" in res.canonical_name


def test_f5_outdated_evidence(classifier):
    """F5: Citing obsolete or superseded evidence for current state assertions."""
    ev = [
        MockPassage(
            doc_id="doc5",
            text="As of 2004, deep learning models are computationally impractical for speech.",
            score=0.75,
            rank=1,
            metadata={"relationship": "SUPPORT", "publication_year": 2004, "is_superseded": True}
        )
    ]
    res = classifier.classify(
        claim_text="Deep learning is computationally impractical for modern speech processing in 2026.",
        verdict="SUPPORT",
        confidence=0.85,
        evidence=ev
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F5
    assert "Outdated Evidence" in res.canonical_name


def test_f6_population_context_mismatch(classifier):
    """F6: Off-domain evidence (e.g. smart grid evidence for quantum FL claim)."""
    ev = [
        MockPassage(
            doc_id="doc6",
            text="Cybersecurity and resilience of smart grids and power distribution systems under attack.",
            score=0.76,
            rank=1,
            metadata={"relationship": "SUPPORT", "topic_match": False, "evidence_eligible": False}
        )
    ]
    res = classifier.classify(
        claim_text="Quantum-resistant federated learning prevents model poisoning.",
        verdict="SUPPORT",
        confidence=0.86,
        evidence=ev,
        question="What are the papers on quantum-resistant federated learning?"
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F6
    assert "Context Mismatch" in res.canonical_name


def test_f7_contradictory_evidence(classifier):
    """F7: Evidence contradicts claim, but verdict is SUPPORT."""
    ev = [
        MockPassage(
            doc_id="doc7",
            text="Empirical results show that the new algorithm degrades performance by 40%.",
            score=0.85,
            rank=1,
            relation="CONTRADICT",
            metadata={"relationship": "CONTRADICT"}
        )
    ]
    res = classifier.classify(
        claim_text="The algorithm dramatically improves performance across all benchmarks.",
        verdict="SUPPORT",
        confidence=0.89,
        evidence=ev
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F7
    assert "Contradictory Evidence" in res.canonical_name
    assert res.severity == "CRITICAL"


def test_f8_evidence_misinterpretation(classifier):
    """F8: Syntactic claim malformation or distortion of evidence."""
    ev = [
        MockPassage(
            doc_id="doc8",
            text="Increasing parameter scale improved accuracy marginally.",
            score=0.72,
            rank=1,
            metadata={"relationship": "SUPPORT"}
        )
    ]
    res = classifier.classify(
        claim_text="Altering increasing model parameter scale directly impacts outcome.",
        verdict="SUPPORT",
        confidence=0.80,
        evidence=ev
    )
    assert res.has_failure is True
    assert res.canonical_code == CanonicalFailureCode.F8
    assert "Misinterpretation" in res.canonical_name


def test_clean_verification_contract_no_failure(classifier):
    """Clean verification: Valid claim + strong matching evidence + correct verdict."""
    ev = [
        MockPassage(
            doc_id="doc9",
            text="Lattice-based cryptography provides post-quantum security guarantees against Shor's algorithm.",
            score=0.88,
            rank=1,
            relation="SUPPORT",
            metadata={"relationship": "SUPPORT", "topic_match": True, "evidence_eligible": True}
        ),
        MockPassage(
            doc_id="doc10",
            text="Theoretical foundations of lattice hard problems remain secure against polynomial quantum search.",
            score=0.86,
            rank=2,
            relation="SUPPORT",
            metadata={"relationship": "SUPPORT", "topic_match": True, "evidence_eligible": True}
        )
    ]
    res = classifier.classify(
        claim_text="Lattice-based cryptography resists Shor's algorithm attacks.",
        verdict="SUPPORT",
        confidence=0.85,
        evidence=ev
    )
    assert res.has_failure is False
    assert res.canonical_code is None
    assert res.severity == "CLEAN"


# ============================================================================
# 3. RUNTIME AUDITOR INTEGRATION
# ============================================================================

def test_runtime_auditor_canonical_fields():
    """RuntimeAuditor must produce records containing canonical_failure_code and name."""
    auditor = RuntimeAuditor()
    # Off-domain claim
    ev = [
        MockPassage(
            doc_id="grid_101",
            text="Smart grid power distribution optimization using SCADA systems.",
            score=0.80,
            rank=1,
            metadata={"topic_match": False, "evidence_eligible": False}
        )
    ]
    record = auditor.audit_claim(
        claim_id="test_claim_01",
        claim_text="Quantum federated learning resists eavesdropping.",
        verdict="SUPPORT",
        confidence=0.90,
        evidence=ev,
        question="Quantum federated learning security"
    )
    assert record.canonical_failure_code == "F6"
    assert "Population/Context Mismatch" in record.canonical_failure_name
    assert record.failure_type == "DOMAIN_MISMATCH"
    assert record.ledger_status in ("RECORDED_HALLUCINATION_LEDGER", "FLAGGED", "LOGGED_TO_AUDIT")


# ============================================================================
# 4. PATTERN MINER INTEGRATION
# ============================================================================

def test_pattern_miner_canonical_fields():
    """All patterns produced by PatternMiner must contain canonical failure code and name."""
    miner = PatternMiner()
    patterns = miner.mine_patterns()
    assert len(patterns) == 9
    for p in patterns:
        assert p.canonical_failure_code in [c.value for c in CanonicalFailureCode]
        assert len(p.canonical_failure_name) > 0


# ============================================================================
# 5. FAILURE MEMORY INTEGRATION & ADAPTIVE VERIFICATION
# ============================================================================

def test_failure_memory_canonical_matching():
    """FailureMemory.match_historical_failures must assign canonical taxonomy."""
    memory = FailureMemory()
    ev = [
        MockPassage(
            doc_id="smart_grid_doc",
            text="Smart grid power system cybersecurity resilience against physical intrusions.",
            score=0.75,
            rank=1,
            metadata={"topic_match": False, "evidence_eligible": False}
        )
    ]
    match_res = memory.match_historical_failures(
        claim_text="Quantum resistant federated learning models secure networks.",
        evidence=ev,
        question="What are the papers on quantum resistant federated learning?",
        preliminary_verdict="SUPPORT",
        confidence=0.90
    )
    assert match_res.matched is True
    assert match_res.canonical_failure_code == "F6"
    assert "Context Mismatch" in match_res.canonical_failure_name

    # Now execute claim recheck
    revised_v, revised_c, filtered_ev, updated_match = memory.execute_claim_recheck(
        claim_text="Quantum resistant federated learning models secure networks.",
        evidence=ev,
        match_result=match_res,
        current_verdict="SUPPORT",
        current_confidence=0.90,
        question="What are the papers on quantum resistant federated learning?"
    )
    assert revised_v == "NEI"
    assert revised_c <= 0.35
    assert len(filtered_ev) == 0
    assert updated_match.claim_recheck_status == "REJECT"
