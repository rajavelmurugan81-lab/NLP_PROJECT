"""Runtime Hallucination Audit Ledger for CiteGuard.

Audits every claim evaluated during live user queries and writes structured records
to results/hallucination_ledger/runtime_audit.jsonl, separating live runtime claim audits
from historical ground-truth benchmark failure datasets.
"""

import os
import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.evaluation.failure_classifier import FailureClassifier
from src.evaluation.failure_taxonomy import CanonicalFailureCode

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_AUDIT_PATH = os.path.join("results", "hallucination_ledger", "runtime_audit.jsonl")


@dataclass
class RuntimeAuditRecord:
    """Audit schema for a single evaluated atomic claim in live user queries."""
    audit_id: str
    run_id: str
    timestamp: str
    question: str
    question_type: str
    claim_id: str
    claim_text: str
    retrieval_mode: str
    openalex_status: str
    local_retrieval_status: str
    evidence_count: int
    supporting_evidence_count: int
    contradicting_evidence_count: int
    irrelevant_evidence_count: int
    best_relevance_score: float
    average_relevance_score: float
    claim_verdict: str
    confidence: float
    failure_type: str  # CLEAN, UNGROUNDED_SUPPORT, UNGROUNDED_CONFIDENCE, DOMAIN_MISMATCH, etc.
    failure_signature: str
    severity: str  # CLEAN, LOW, MEDIUM, HIGH, CRITICAL
    ledger_status: str  # CLEAN_CONTRACT, LOGGED_TO_AUDIT
    explanation: str
    canonical_failure_code: str = "NONE"  # F1 - F8 or NONE
    canonical_failure_name: str = "Clean Verification Contract"
    matched_pattern_ids: List[str] = field(default_factory=list)
    pattern_match_score: float = 0.0
    risk_level: str = "LOW"
    adaptive_action: str = "NONE"
    prevention_triggered: bool = False
    claim_recheck_status: str = "ACCEPT"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeAuditor:
    """Audits runtime verification outcomes against verification and retrieval rules."""

    def __init__(self, ledger_filepath: str = DEFAULT_RUNTIME_AUDIT_PATH):
        self.ledger_filepath = ledger_filepath
        os.makedirs(os.path.dirname(self.ledger_filepath), exist_ok=True)
        self.classifier = FailureClassifier()

    def audit_claim(
        self,
        claim_id: str,
        claim_text: str,
        verdict: str,
        confidence: float,
        evidence: List[Any],
        question: str,
        question_type: str = "SCIENTIFIC_HYPOTHESIS",
        retrieval_mode: str = "openalex_hybrid",
        openalex_status: str = "CONNECTED_WITH_RESULTS",
        local_retrieval_status: str = "READY",
        run_id: str = "run_live",
        matched_pattern_ids: Optional[List[str]] = None,
        pattern_match_score: float = 0.0,
        risk_level: str = "LOW",
        adaptive_action: str = "NONE",
        prevention_triggered: bool = False,
        claim_recheck_status: str = "ACCEPT",
        failure_type_override: Optional[str] = None,
        canonical_code_override: Optional[str] = None
    ) -> RuntimeAuditRecord:
        """Audit a verified claim, detect failure signatures, and log to runtime_audit.jsonl."""
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        t_now = datetime.now(timezone.utc).isoformat()

        # Parse evidence items
        scores = []
        supp_count = 0
        contra_count = 0
        irrel_count = 0

        for ev in evidence:
            sc = float(getattr(ev, "score", 0.0) or (ev.get("score") if isinstance(ev, dict) else 0.0) or 0.0)
            rel = str(getattr(ev, "relation", "") or (ev.get("relation") if isinstance(ev, dict) else "") or "").upper()
            rel_label = str(getattr(ev, "metadata", {}).get("relationship", "") if hasattr(ev, "metadata") else (ev.get("metadata", {}).get("relationship", "") if isinstance(ev, dict) else "")).upper()
            
            scores.append(sc)
            if "SUPPORT" in rel or "SUPPORT" in rel_label:
                supp_count += 1
            elif "CONTRADICT" in rel or "CONTRADICT" in rel_label or "REFUTE" in rel:
                contra_count += 1
            elif sc < 0.50:
                irrel_count += 1

        best_score = round(max(scores), 4) if scores else 0.0
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        v_norm = str(verdict).upper().replace(" ", "_")
        if "." in v_norm:
            v_norm = v_norm.split(".")[-1]

        # Execute Canonical Classification via FailureClassifier
        cls_res = self.classifier.classify(
            claim_text=claim_text,
            evidence=evidence,
            verdict=v_norm,
            confidence=confidence,
            question=question,
            question_type=question_type
        )

        canonical_code_str = canonical_code_override or (cls_res.canonical_code.value if cls_res.canonical_code else "NONE")
        canonical_name_str = cls_res.canonical_name

        if failure_type_override:
            failure_type = failure_type_override
            failure_signature = f"FS-{failure_type_override}"
            severity = "HIGH"
            ledger_status = "LOGGED_TO_AUDIT"
            explanation = f"Independently adjudicated failure recorded under category {failure_type_override}."
        elif cls_res.has_failure:
            failure_type = cls_res.legacy_failure_type
            failure_signature = cls_res.legacy_failure_signature
            severity = cls_res.severity
            ledger_status = "FLAGGED" if severity == "HIGH" and failure_type == "DOMAIN_MISMATCH" else "LOGGED_TO_AUDIT"
            explanation = cls_res.reason
        else:
            failure_type = "NONE"
            failure_signature = "NONE"
            severity = "NONE"
            ledger_status = "CLEAN"
            explanation = cls_res.reason

        # Create record
        record = RuntimeAuditRecord(
            audit_id=audit_id,
            run_id=run_id,
            timestamp=t_now,
            question=question,
            question_type=question_type,
            claim_id=claim_id,
            claim_text=claim_text,
            retrieval_mode=retrieval_mode,
            openalex_status=openalex_status,
            local_retrieval_status=local_retrieval_status,
            evidence_count=len(evidence),
            supporting_evidence_count=supp_count,
            contradicting_evidence_count=contra_count,
            irrelevant_evidence_count=irrel_count,
            best_relevance_score=best_score,
            average_relevance_score=avg_score,
            claim_verdict=v_norm,
            confidence=confidence,
            failure_type=failure_type,
            failure_signature=failure_signature,
            severity=severity,
            ledger_status=ledger_status,
            explanation=explanation,
            canonical_failure_code=canonical_code_str,
            canonical_failure_name=canonical_name_str,
            matched_pattern_ids=matched_pattern_ids or [],
            pattern_match_score=pattern_match_score,
            risk_level=risk_level,
            adaptive_action=adaptive_action,
            prevention_triggered=prevention_triggered,
            claim_recheck_status=claim_recheck_status,
            metadata={
                "scores": scores,
                "diagnostic_signals": cls_res.diagnostic_signals
            }
        )

        # Persist to disk
        self._append_to_file(record)
        return record

    def _append_to_file(self, record: RuntimeAuditRecord):
        """Append record to runtime audit jsonl file."""
        try:
            with open(self.ledger_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"[RuntimeAuditor] Failed to write record to {self.ledger_filepath}: {e}")

    def load_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load records from runtime audit ledger."""
        if not os.path.exists(self.ledger_filepath):
            return []
        records = []
        try:
            with open(self.ledger_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if limit:
                return records[-limit:]
            return records
        except Exception as e:
            logger.warning(f"[RuntimeAuditor] Failed to read {self.ledger_filepath}: {e}")
            return []
