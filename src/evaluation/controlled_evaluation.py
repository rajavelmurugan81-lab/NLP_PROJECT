"""Controlled Evaluation Suite for Canonical F1-F8 Scientific Failure Taxonomy.

Systematically evaluates CiteGuard's failure classification and objection mechanisms
against ground-truth controlled failure scenarios spanning F1 to F8:
- F1: Unsupported Claim
- F2: Citation Mismatch
- F3: Overgeneralization
- F4: Causal Leap
- F5: Outdated Evidence
- F6: Population / Context Mismatch
- F7: Contradictory Evidence
- F8: Evidence Misinterpretation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import logging

from src.evaluation.failure_taxonomy import CanonicalFailureType, CanonicalFailureCode, CANONICAL_FAILURE_REGISTRY
from src.evaluation.failure_classifier import FailureClassifier
from src.verification.schema import RetrievedEvidence

logger = logging.getLogger(__name__)


@dataclass
class ControlledFailureCase:
    """A benchmark test case representing an intentional, controlled failure mode."""
    case_id: str
    expected_failure_type: CanonicalFailureType
    claim_text: str
    evidence: List[RetrievedEvidence]
    preliminary_verdict: str
    confidence: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlledEvaluationReport:
    """Evaluation metrics summary across controlled F1-F8 failure scenarios."""
    total_test_cases: int
    detected_count: int
    detection_rate: float
    exact_match_count: int
    exact_match_accuracy: float
    per_category_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ControlledEvaluationSuite:
    """Harness for testing F1-F8 failure detection and classification accuracy."""

    def __init__(self, classifier: Optional[FailureClassifier] = None):
        self.classifier = classifier or FailureClassifier()

    def generate_controlled_cases(self) -> List[ControlledFailureCase]:
        """Construct canonical controlled evaluation cases for F1 through F8."""
        cases = [
            # F1: Unsupported Claim (zero evidence provided)
            ControlledFailureCase(
                case_id="ctrl_f1_001",
                expected_failure_type=CanonicalFailureType.F1,
                claim_text="Exogenous administration of substance Z extends human lifespan by 30 years.",
                evidence=[],
                preliminary_verdict="SUPPORTED",
                confidence=0.90,
                description="Claim asserts lifespan extension with empty retrieved evidence."
            ),
            # F1: Unsupported Claim (evidence completely irrelevant)
            ControlledFailureCase(
                case_id="ctrl_f1_002",
                expected_failure_type=CanonicalFailureType.F1,
                claim_text="Quantum annealing achieves polynomial time factorisation of RSA integers.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="bio_doc",
                        text="MicroRNA-21 suppresses cell apoptosis in epithelial ovarian cancer lines.",
                        score=0.10,
                        rank=1
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.85,
                description="Claim asserts quantum factorization with biologically irrelevant evidence."
            ),
            # F2: Citation Mismatch (cross-paper / wrong target DOI)
            ControlledFailureCase(
                case_id="ctrl_f2_001",
                expected_failure_type=CanonicalFailureType.F2,
                claim_text="Brown et al. (2020) demonstrated few-shot learning capabilities in GPT-3.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="p_vgg_001",
                        text="Very deep convolutional networks achieve top-5 error of 7.3% on ImageNet.",
                        title="Very Deep Convolutional Networks",
                        score=0.82,
                        rank=1,
                        metadata={"url_doi": "10.48550/arXiv.1409.1556", "authors": "Simonyan and Zisserman"}
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.88,
                description="GPT-3 claim backed by ImageNet VGG citation."
            ),
            # F3: Overgeneralization (universal quantifier "in all humans" from limited sample)
            ControlledFailureCase(
                case_id="ctrl_f3_001",
                expected_failure_type=CanonicalFailureType.F3,
                claim_text="Sleep deprivation permanently destroys cognitive function across all human populations worldwide.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="sleep_study",
                        text="In a trial of 24 healthy male college students, 36 hours of sleep deprivation reduced vigilance.",
                        score=0.91,
                        rank=1
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.92,
                description="Universal claim extrapolated from small cohort study of 24 subjects."
            ),
            # F4: Causal Leap (correlation in evidence -> causal assertion in claim)
            ControlledFailureCase(
                case_id="ctrl_f4_001",
                expected_failure_type=CanonicalFailureType.F4,
                claim_text="High dietary sodium intake directly causes hypertension and myocardial infarction.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="sodium_corr",
                        text="High sodium intake is statistically associated and correlated with elevated systolic blood pressure.",
                        score=0.89,
                        rank=1
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.87,
                description="Direct causal assertion from purely observational correlation."
            ),
            # F5: Outdated Evidence (historical assertion superseded by newer research)
            ControlledFailureCase(
                case_id="ctrl_f5_001",
                expected_failure_type=CanonicalFailureType.F5,
                claim_text="Peptic ulcers are predominantly caused by psychological stress and dietary acid.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="ulcer_old",
                        text="Peptic ulcers are historical conditions associated with occupational stress and gastric acid secretion.",
                        score=0.80,
                        rank=1,
                        metadata={"publication_year": 1978}
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.85,
                description="Superseded medical belief prior to Helicobacter pylori discovery."
            ),
            # F6: Population / Context Mismatch (in vitro cell line -> human clinical claim)
            ControlledFailureCase(
                case_id="ctrl_f6_001",
                expected_failure_type=CanonicalFailureType.F6,
                claim_text="Curcumin cures aggressive glioblastoma in human clinical cancer patients.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="curcumin_vitro",
                        text="In vitro cell culture assays demonstrate curcumin induces apoptosis in U87 glioblastoma cell lines.",
                        score=0.88,
                        rank=1
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.90,
                description="In vitro petri-dish findings extrapolated directly to human clinical cure."
            ),
            # F7: Contradictory Evidence (conflicting findings among evidence passages)
            ControlledFailureCase(
                case_id="ctrl_f7_001",
                expected_failure_type=CanonicalFailureType.F7,
                claim_text="Daily moderate caffeine consumption improves cardiovascular health.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="ev_pos",
                        text="Moderate caffeine consumption reduces all-cause cardiovascular mortality by 12%.",
                        score=0.92,
                        rank=1
                    ),
                    RetrievedEvidence(
                        doc_id="ev_neg",
                        text="Caffeine intake shows no protective benefit and significantly elevates arterial stiffness.",
                        score=0.90,
                        rank=2
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.85,
                description="Direct contradiction between retrieved cardiovascular outcomes."
            ),
            # F8: Evidence Misinterpretation (inverted directional findings)
            ControlledFailureCase(
                case_id="ctrl_f8_001",
                expected_failure_type=CanonicalFailureType.F8,
                claim_text="Chronic intermittent hypoxia significantly increases executive cognitive function.",
                evidence=[
                    RetrievedEvidence(
                        doc_id="hypoxia_study",
                        text="Chronic intermittent hypoxia impairs and reduces executive cognitive function in murine models.",
                        score=0.94,
                        rank=1
                    )
                ],
                preliminary_verdict="SUPPORTED",
                confidence=0.89,
                description="Claim asserts an increase when the literature clearly demonstrates impairment/reduction."
            )
        ]
        return cases

    def run_evaluation(self, cases: Optional[List[ControlledFailureCase]] = None) -> ControlledEvaluationReport:
        """Execute evaluation across the controlled failure scenarios."""
        test_cases = cases or self.generate_controlled_cases()
        total = len(test_cases)
        detected = 0
        exact_matches = 0

        per_cat: Dict[str, Dict[str, Any]] = {}
        for code_enum, defn in CANONICAL_FAILURE_REGISTRY.items():
            per_cat[code_enum.value] = {
                "name": defn.name,
                "total": 0,
                "detected": 0,
                "exact_matches": 0
            }

        for case in test_cases:
            code = case.expected_failure_type.value
            per_cat[code]["total"] += 1

            sig = self.classifier.classify_failure(
                claim_text=case.claim_text,
                evidence=case.evidence,
                predicted_label=case.preliminary_verdict,
                confidence=case.confidence,
                claim_id=case.case_id
            )

            if sig and sig.failure_type:
                detected += 1
                per_cat[code]["detected"] += 1

                # Check if predicted canonical type matches expected canonical type
                pred_type = getattr(sig, "canonical_type", None) or getattr(sig, "failure_type", None)
                if pred_type == case.expected_failure_type or (hasattr(pred_type, "code") and pred_type.code == code):
                    exact_matches += 1
                    per_cat[code]["exact_matches"] += 1
                elif hasattr(sig, "canonical_code") and sig.canonical_code == code:
                    exact_matches += 1
                    per_cat[code]["exact_matches"] += 1

        det_rate = round(detected / total, 4) if total > 0 else 0.0
        acc_rate = round(exact_matches / total, 4) if total > 0 else 0.0

        for k, v in per_cat.items():
            tot = v["total"]
            v["detection_rate"] = round(v["detected"] / tot, 4) if tot > 0 else 0.0
            v["accuracy"] = round(v["exact_matches"] / tot, 4) if tot > 0 else 0.0

        report = ControlledEvaluationReport(
            total_test_cases=total,
            detected_count=detected,
            detection_rate=det_rate,
            exact_match_count=exact_matches,
            exact_match_accuracy=acc_rate,
            per_category_metrics=per_cat
        )
        return report
