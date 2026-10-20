"""Deterministic Failure Analyzer for classifying retrieval and reasoning failure modes."""

from typing import List, Dict, Any, Optional, Set
from src.evaluation.failure_taxonomy import FailureType
from src.evaluation.failure_signature import FailureSignature


def classify_retrieval_failure(
    ground_truth_doc_ids: Set[str],
    retrieved_doc_ids: List[str]
) -> bool:
    """Determine if a retrieval failure occurred.

    Returns True if ground-truth evidence exists in corpus but none was retrieved in top-K.
    """
    if not ground_truth_doc_ids:
        return False  # No ground truth evidence specified
    top_k_set = set(retrieved_doc_ids)
    return len(top_k_set.intersection(ground_truth_doc_ids)) == 0


def classify_reasoning_failure(
    ground_truth_doc_ids: Set[str],
    retrieved_doc_ids: List[str],
    predicted_label: str,
    ground_truth_label: str
) -> bool:
    """Determine if a reasoning failure occurred.

    Returns True if ground-truth evidence WAS successfully retrieved, but predicted verdict is wrong.
    """
    if not ground_truth_doc_ids:
        return False
    if predicted_label.upper() == ground_truth_label.upper():
        return False

    top_k_set = set(retrieved_doc_ids)
    evidence_retrieved = len(top_k_set.intersection(ground_truth_doc_ids)) > 0
    return evidence_retrieved and (predicted_label.upper() != ground_truth_label.upper())


class FailureAnalyzer:
    """Deterministic analyzer converting evaluation instances into FailureSignatures."""

    def __init__(self, default_run_id: str = "run_eval"):
        self.default_run_id = default_run_id

    def analyze_instance(
        self,
        claim_id: str,
        claim_text: str,
        predicted_label: str,
        ground_truth_label: str,
        retrieved_results: List[Dict[str, Any]],
        ground_truth_doc_ids: Optional[Set[str]] = None,
        dataset: str = "unknown",
        split: str = "train",
        retrieval_mode: str = "hybrid",
        run_id: Optional[str] = None
    ) -> Optional[FailureSignature]:
        """Analyze a prediction instance. Returns FailureSignature if failed, None if correct."""
        pred_norm = predicted_label.upper().strip()
        gt_norm = ground_truth_label.upper().strip()

        # If prediction matches ground truth, no failure occurred
        if pred_norm == gt_norm:
            return None

        retrieved_ids = [r["doc_id"] for r in retrieved_results]
        retrieved_ranks = [r.get("rank", i + 1) for i, r in enumerate(retrieved_results)]
        gt_set = set(ground_truth_doc_ids) if ground_truth_doc_ids else set()

        evidence_available = len(gt_set) > 0
        evidence_retrieved = len(gt_set.intersection(set(retrieved_ids))) > 0 if gt_set else False

        # Deterministic Failure Classification Rules
        if not retrieved_ids:
            failure_type = FailureType.NO_EVIDENCE
            explanation = "No evidence documents were returned by the retrieval engine."
        elif gt_set and classify_retrieval_failure(gt_set, retrieved_ids):
            failure_type = FailureType.RETRIEVAL_FAILURE
            explanation = f"Ground-truth evidence IDs {list(gt_set)} exist but were not retrieved in top-{len(retrieved_ids)}."
        elif gt_set and classify_reasoning_failure(gt_set, retrieved_ids, pred_norm, gt_norm):
            failure_type = FailureType.REASONING_FAILURE
            explanation = f"Ground-truth evidence was retrieved, but predicted verdict '{pred_norm}' disagrees with ground truth '{gt_norm}'."
        elif pred_norm == "NEUTRAL" and gt_norm in ["SUPPORT", "REFUTE"]:
            failure_type = FailureType.INSUFFICIENT_EVIDENCE
            explanation = f"System returned NEUTRAL due to insufficient evidence support for ground truth '{gt_norm}'."
        elif (pred_norm == "SUPPORT" and gt_norm == "REFUTE") or (pred_norm == "REFUTE" and gt_norm == "SUPPORT"):
            failure_type = FailureType.CONTRADICTORY_EVIDENCE
            explanation = f"Direct verdict contradiction: predicted '{pred_norm}' vs ground truth '{gt_norm}'."
        elif pred_norm == "PARTIALLY_SUPPORTED":
            failure_type = FailureType.PARTIAL_EVIDENCE
            explanation = "Evidence supports only part of the scientific claim."
        else:
            failure_type = FailureType.UNKNOWN_FAILURE
            explanation = f"Unclassified mismatch: predicted '{pred_norm}' vs ground truth '{gt_norm}'."

        return FailureSignature(
            claim_id=claim_id,
            claim_text=claim_text,
            dataset=dataset,
            split=split,
            predicted_label=pred_norm,
            ground_truth_label=gt_norm,
            failure_type=failure_type,
            retrieved_document_ids=retrieved_ids,
            retrieved_ranks=retrieved_ranks,
            retrieval_mode=retrieval_mode,
            evidence_available=evidence_available,
            evidence_retrieved=evidence_retrieved,
            explanation=explanation,
            run_id=run_id or self.default_run_id
        )

    def analyze_runtime_instance(
        self,
        claim_id: str,
        claim_text: str,
        predicted_label: str,
        confidence: float,
        evidence: List[Any],
        disagreement_status: bool = False,
        resolution_method: str = "direct_consensus",
        dataset: str = "research_question",
        retrieval_mode: str = "hybrid",
        run_id: Optional[str] = None,
        question: Optional[str] = None
    ) -> Optional[FailureSignature]:
        """Analyze a runtime verification instance for live research questions without ground truth labels."""
        pred_norm = predicted_label.upper().strip()

        def _get_doc_id(item: Any) -> str:
            if hasattr(item, "doc_id"):
                return str(item.doc_id)
            if isinstance(item, dict):
                return str(item.get("doc_id", ""))
            return str(item)

        def _get_score(item: Any) -> float:
            if hasattr(item, "score"):
                return float(item.score)
            if isinstance(item, dict):
                return float(item.get("score", item.get("rrf_score", 0.0)))
            return 0.0

        def _get_rank(item: Any, default_rank: int) -> int:
            if hasattr(item, "rank"):
                return int(item.rank)
            if isinstance(item, dict):
                return int(item.get("rank", default_rank))
            return default_rank

        retrieved_ids = [_get_doc_id(e) for e in evidence] if evidence else []
        retrieved_ranks = [_get_rank(e, i + 1) for i, e in enumerate(evidence)] if evidence else []

        failure_type: Optional[FailureType] = None
        failure_subtype: Optional[str] = None
        explanation: str = ""
        severity: str = "MEDIUM"

        # Check 1: NO_EVIDENCE
        if not evidence or len(evidence) == 0:
            failure_type = FailureType.NO_EVIDENCE
            explanation = "No relevant scientific evidence passages were retrieved from the corpus."
            severity = "HIGH"

        # Check 2: UNGROUNDED_CONFIDENCE (High confidence assigned to NEI / low evidence)
        elif confidence >= 0.85 and (pred_norm in ("NEI", "INSUFFICIENT_EVIDENCE") or len(evidence) < 2):
            failure_type = FailureType.INSUFFICIENT_EVIDENCE
            failure_subtype = "UNGROUNDED_CONFIDENCE"
            explanation = f"High confidence ({confidence*100:.0f}%) assigned despite insufficient evidence support ({pred_norm})."
            severity = "MEDIUM"

        # Check 3: AGENT_DISAGREEMENT
        elif disagreement_status or resolution_method == "debate_resolution":
            failure_type = FailureType.REASONING_FAILURE
            failure_subtype = "AGENT_DISAGREEMENT"
            explanation = "Multi-agent consensus split; required debate resolution."
            severity = "MEDIUM"

        # Check 4: CONTRADICTORY_EVIDENCE / CLAIM_EVIDENCE_MISMATCH
        elif pred_norm in ("REFUTED", "REFUTE"):
            failure_type = FailureType.CONTRADICTORY_EVIDENCE
            failure_subtype = "CLAIM_EVIDENCE_MISMATCH"
            explanation = "Retrieved evidence contradicts or refutes the generated scientific claim hypothesis."
            severity = "HIGH"

        # Check 5: LOW_EVIDENCE_SUPPORT
        elif len(evidence) == 0 or all(_get_score(e) <= 0.0 for e in evidence):
            failure_type = FailureType.INSUFFICIENT_EVIDENCE
            failure_subtype = "LOW_EVIDENCE_SUPPORT"
            explanation = f"Low evidence coverage: retrieved only {len(evidence)} passages with non-positive relevance scores."
            severity = "LOW"

        if failure_type is None:
            return None  # Clean execution contract

        sig = FailureSignature(
            claim_id=claim_id,
            claim_text=claim_text,
            dataset=dataset,
            split="live",
            predicted_label=pred_norm,
            ground_truth_label="NONE",
            failure_type=failure_type,
            failure_subtype=failure_subtype,
            retrieved_document_ids=retrieved_ids,
            retrieved_ranks=retrieved_ranks,
            retrieval_mode=retrieval_mode,
            evidence_available=len(evidence) > 0,
            evidence_retrieved=len(evidence) > 0,
            confidence=confidence,
            explanation=explanation,
            run_id=run_id or self.default_run_id,
            metadata={"severity": severity, "evidence_count": len(evidence), "question": question or ""}
        )
        return sig
