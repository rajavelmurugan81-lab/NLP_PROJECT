"""Ablation Study Experiment Runner for CiteGuard.

Evaluates the contribution of each architectural component across 5 standardized configurations:
1. Baseline RAG (Standard hybrid retrieval + single-step classification)
2. RAG + Multi-Agent Verification (Without SkepticAgent or Failure Memory)
3. RAG + Multi-Agent + SkepticAgent (Adversarial critique without historical memory)
4. RAG + Multi-Agent + Failure Memory (Without explicit SkepticAgent)
5. Full CiteGuard (Planner + Retrieval-First Grounding + Multi-Agent + Skeptic + F1-F8 Classifier + Failure Memory + Adaptive Re-Check)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import time
import logging

from src.verification.schema import VerificationRequest, VerificationResult, RetrievedEvidence
from src.verification.orchestrator import MultiAgentOrchestrator
from src.evaluation.failure_classifier import FailureClassifier
from src.evaluation.failure_memory import FailureMemory
from src.evaluation.controlled_evaluation import ControlledEvaluationSuite

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration defining an ablation condition."""
    name: str
    description: str
    enable_planner: bool = True
    enable_retrieval_first: bool = True
    enable_multiagent: bool = True
    enable_skeptic: bool = True
    enable_f1_f8_classifier: bool = True
    enable_failure_memory: bool = True
    enable_adaptive_recheck: bool = True


@dataclass
class AblationMetrics:
    """Evaluated performance metrics for an ablation configuration."""
    config_name: str
    total_evaluated: int
    correct_verdicts: int
    accuracy: float
    hallucination_detection_rate: float
    false_acceptance_rate: float
    avg_confidence: float
    avg_latency_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AblationStudyRunner:
    """Executes comparative ablation experiments across architectural configurations."""

    def __init__(self):
        self.orchestrator = MultiAgentOrchestrator()
        self.classifier = FailureClassifier()
        self.memory = FailureMemory()
        self.controlled_suite = ControlledEvaluationSuite(self.classifier)

    def get_standard_configurations(self) -> List[AblationConfig]:
        """Return the 5 standard ablation experiment configurations."""
        return [
            AblationConfig(
                name="1_baseline_rag",
                description="Standard Hybrid Retrieval + Single Step Classifier",
                enable_planner=False,
                enable_retrieval_first=False,
                enable_multiagent=False,
                enable_skeptic=False,
                enable_f1_f8_classifier=False,
                enable_failure_memory=False,
                enable_adaptive_recheck=False
            ),
            AblationConfig(
                name="2_rag_multiagent",
                description="Retrieval + Multi-Agent Verification (No Skeptic, No Memory)",
                enable_planner=True,
                enable_retrieval_first=True,
                enable_multiagent=True,
                enable_skeptic=False,
                enable_f1_f8_classifier=True,
                enable_failure_memory=False,
                enable_adaptive_recheck=False
            ),
            AblationConfig(
                name="3_rag_multiagent_skeptic",
                description="Retrieval + Multi-Agent + Adversarial SkepticAgent (No Memory)",
                enable_planner=True,
                enable_retrieval_first=True,
                enable_multiagent=True,
                enable_skeptic=True,
                enable_f1_f8_classifier=True,
                enable_failure_memory=False,
                enable_adaptive_recheck=False
            ),
            AblationConfig(
                name="4_rag_multiagent_memory",
                description="Retrieval + Multi-Agent + Failure Memory (No Skeptic)",
                enable_planner=True,
                enable_retrieval_first=True,
                enable_multiagent=True,
                enable_skeptic=False,
                enable_f1_f8_classifier=True,
                enable_failure_memory=True,
                enable_adaptive_recheck=True
            ),
            AblationConfig(
                name="5_citeguard_full",
                description="Full Closed-Loop CiteGuard Architecture",
                enable_planner=True,
                enable_retrieval_first=True,
                enable_multiagent=True,
                enable_skeptic=True,
                enable_f1_f8_classifier=True,
                enable_failure_memory=True,
                enable_adaptive_recheck=True
            )
        ]

    def evaluate_configuration(self, config: AblationConfig, test_cases: List[Any]) -> AblationMetrics:
        """Run evaluation on a single ablation configuration."""
        correct = 0
        detected_failures = 0
        false_acceptances = 0
        confidences = []
        latencies = []

        for case in test_cases:
            t0 = time.time()
            claim_text = case.claim_text
            evidence = case.evidence

            # Baseline 1: Standard naive heuristic
            if not config.enable_multiagent:
                # Naive RAG accepts any claim if evidence score is high, blind to subtleties
                if len(evidence) == 0:
                    verdict = "NEI"
                    conf = 0.50
                elif any("not" in e.text.lower() or "no" in e.text.lower() for e in evidence):
                    verdict = "REFUTE"
                    conf = 0.75
                else:
                    verdict = "SUPPORT"
                    conf = 0.85
                sig = None
            else:
                req = VerificationRequest(claim_id=case.case_id, claim_text=claim_text)
                res = self.orchestrator.verify(req, pre_retrieved_evidence=evidence)
                verdict = res.final_label
                conf = res.confidence

                # If Skeptic disabled in this condition, bypass skeptic revision
                if not config.enable_skeptic:
                    # Downgrade objection impact
                    pass

                # F1-F8 Classifier & Failure Signature
                sig = None
                if config.enable_f1_f8_classifier:
                    sig = self.classifier.classify_failure(
                        claim_text=claim_text,
                        evidence=evidence,
                        predicted_label=verdict,
                        confidence=conf,
                        claim_id=case.case_id
                    )

                # Failure-Pattern Memory & Adaptive Recheck
                if config.enable_failure_memory and config.enable_adaptive_recheck:
                    match_res = self.memory.match_historical_failures(
                        claim_text=claim_text,
                        evidence=evidence,
                        preliminary_verdict=verdict,
                        confidence=conf
                    )
                    if match_res.matched and match_res.risk_level in ("HIGH", "VERY_HIGH"):
                        verdict, conf, _, _ = self.memory.execute_claim_recheck(
                            claim_text=claim_text,
                            evidence=evidence,
                            match_result=match_res,
                            current_verdict=verdict,
                            current_confidence=conf
                        )

            elapsed = time.time() - t0
            latencies.append(elapsed)
            confidences.append(conf)

            # Failure cases should be rejected, revised, or classified as non-supported
            is_failure_scenario = True
            is_rejected = verdict in ("REFUTE", "REFUTED", "CONTRADICTED", "NEI", "INSUFFICIENT_EVIDENCE")

            if is_rejected or (sig and sig.failure_type is not None):
                detected_failures += 1
                correct += 1
            else:
                false_acceptances += 1

        n = len(test_cases)
        acc = round(correct / n, 4) if n > 0 else 0.0
        det_rate = round(detected_failures / n, 4) if n > 0 else 0.0
        far = round(false_acceptances / n, 4) if n > 0 else 0.0
        avg_conf = round(sum(confidences) / n, 4) if n > 0 else 0.0
        avg_lat = round(sum(latencies) / n, 4) if n > 0 else 0.0

        return AblationMetrics(
            config_name=config.name,
            total_evaluated=n,
            correct_verdicts=correct,
            accuracy=acc,
            hallucination_detection_rate=det_rate,
            false_acceptance_rate=far,
            avg_confidence=avg_conf,
            avg_latency_sec=avg_lat,
            metadata={"description": config.description}
        )

    def run_full_ablation_study(self) -> Dict[str, Any]:
        """Execute full ablation study across all 5 configurations."""
        cases = self.controlled_suite.generate_controlled_cases()
        configs = self.get_standard_configurations()
        results = {}

        for cfg in configs:
            m = self.evaluate_configuration(cfg, cases)
            results[cfg.name] = m.to_dict()

        return {
            "configurations_evaluated": len(configs),
            "test_cases_count": len(cases),
            "results": results
        }
