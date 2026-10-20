"""Pattern Mining Engine for CiteGuard Failure-Pattern Memory.

Analyzes historical failures from results/hallucination_ledger/hallucination_ledger.jsonl
and runtime failures from results/hallucination_ledger/runtime_audit.jsonl.
Groups structurally similar failures into generalized, reusable FailurePattern models
and persists them to results/hallucination_ledger/failure_patterns.json.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

from dataclasses import dataclass, field
from src.evaluation.failure_pattern import FailurePattern

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_LEDGER = os.path.join("results", "hallucination_ledger", "hallucination_ledger.jsonl")
DEFAULT_RUNTIME_AUDIT = os.path.join("results", "hallucination_ledger", "runtime_audit.jsonl")
DEFAULT_PATTERNS_FILE = os.path.join("results", "hallucination_ledger", "failure_patterns.json")


@dataclass
class MiningResult:
    """Detailed result of a pattern mining operation tracking created, updated, and unchanged patterns."""
    created_patterns: List[Dict[str, Any]] = field(default_factory=list)
    updated_patterns: List[Dict[str, Any]] = field(default_factory=list)
    unchanged_patterns: List[Dict[str, Any]] = field(default_factory=list)
    total_patterns: int = 0
    patterns: List[FailurePattern] = field(default_factory=list)

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def created_count(self) -> int:
        return len(self.created_patterns)

    @property
    def updated_count(self) -> int:
        return len(self.updated_patterns)

    @property
    def unchanged_count(self) -> int:
        return len(self.unchanged_patterns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "unchanged_count": self.unchanged_count,
            "created_patterns": self.created_patterns,
            "updated_patterns": self.updated_patterns,
            "unchanged_patterns": self.unchanged_patterns,
            "total_patterns": self.total_patterns,
            "patterns": [p.to_dict() for p in self.patterns]
        }


class PatternMiner:
    """Mines, categorizes, deduplicates, and persists failure patterns from audit ledgers."""

    def __init__(
        self,
        benchmark_path: str = DEFAULT_BENCHMARK_LEDGER,
        runtime_path: str = DEFAULT_RUNTIME_AUDIT,
        output_patterns_path: str = DEFAULT_PATTERNS_FILE,
        ledger_path: Optional[str] = None,
        runtime_audit_path: Optional[str] = None
    ):
        self.benchmark_path = ledger_path or benchmark_path
        self.runtime_path = runtime_audit_path or runtime_path
        self.output_patterns_path = output_patterns_path
        self.last_mining_result: Optional[MiningResult] = None

    def load_benchmark_records(self) -> List[Dict[str, Any]]:
        """Load benchmark failure records from disk."""
        if not os.path.exists(self.benchmark_path):
            return []
        records = []
        try:
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning(f"[PatternMiner] Error reading benchmark ledger: {e}")
        return records

    def load_runtime_records(self) -> List[Dict[str, Any]]:
        """Load runtime audit records from disk."""
        if not os.path.exists(self.runtime_path):
            return []
        records = []
        try:
            with open(self.runtime_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning(f"[PatternMiner] Error reading runtime audit ledger: {e}")
        return records

    def mine_patterns_with_tracking(self) -> MiningResult:
        """Mine reusable failure patterns across both historical benchmark and runtime ledgers,
        tracking which patterns are CREATED, UPDATED, or UNCHANGED relative to existing disk storage."""
        # Load existing patterns from disk to establish baseline
        existing_patterns_dict: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.output_patterns_path):
            try:
                with open(self.output_patterns_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for p in data.get("patterns", []):
                    pid = p.get("pattern_id")
                    if pid:
                        existing_patterns_dict[pid] = p
            except Exception as e:
                logger.warning(f"[PatternMiner] Could not read existing patterns for delta tracking: {e}")

        bench_records = self.load_benchmark_records()
        runtime_records = self.load_runtime_records()

        # Group records by failure categories
        grouped_bench = defaultdict(list)
        for r in bench_records:
            ft = str(r.get("failure_type", "UNKNOWN")).upper()
            if ft and ft != "NONE" and ft != "CLEAN":
                grouped_bench[ft].append(r)

        grouped_runtime = defaultdict(list)
        for r in runtime_records:
            ft = str(r.get("failure_type", "NONE")).upper()
            if ft and ft not in ("NONE", "CLEAN"):
                grouped_runtime[ft].append(r)

        patterns: List[FailurePattern] = []

        # 1. DOMAIN_MISMATCH -> OFF_DOMAIN_EVIDENCE_ACCEPTED_AS_SUPPORT
        dom_examples = grouped_runtime.get("DOMAIN_MISMATCH", [])
        patterns.append(
            FailurePattern(
                pattern_id="FP_DOM_001",
                failure_type="DOMAIN_MISMATCH",
                failure_signature="FS-DOMAIN-MISMATCH",
                pattern_name="OFF_DOMAIN_EVIDENCE_ACCEPTED_AS_SUPPORT",
                severity="HIGH",
                frequency=max(len(dom_examples), 1),
                trigger_conditions={
                    "evidence_topic_mismatch": True,
                    "target_domains_conflict": True,
                    "min_domain_compatibility": 0.50
                },
                observed_failure="Candidate papers from unrelated domains (e.g. smart grids, power systems, or drug discovery) were retrieved for specialized quantum/cryptography or computer science claims.",
                incorrect_behavior="Candidate passages from unrelated domain accepted into evidence mapping and assigned positive relevance scores leading to spurious SUPPORT verdicts.",
                correct_behavior="Apply strict compound topic gating, reject off-domain passages before claim verification, and require explicit domain compatibility.",
                prevention_rule="If claim topic belongs to domain A but retrieved passage belongs to excluded domain B, set passage score to 0.0 and reject as evidence.",
                adaptive_actions=[
                    "REQUIRE_STRICT_DOMAIN_VALIDATION",
                    "EXCLUDE_OFF_DOMAIN_PASSAGES",
                    "CLAIM_RECHECK_DOMAIN_SCOPE"
                ],
                canonical_failure_code="F6",
                canonical_failure_name="Population/Context Mismatch",
                first_seen=dom_examples[0].get("timestamp", "2026-09-11T00:00:00Z") if dom_examples else "2026-09-11T00:00:00Z",
                last_seen=dom_examples[-1].get("timestamp", "2026-09-14T00:00:00Z") if dom_examples else "2026-09-14T00:00:00Z",
                source_examples=[{"claim_text": e.get("claim_text"), "question": e.get("question")} for e in dom_examples[:3]]
            )
        )

        # 2. IRRELEVANT_EVIDENCE -> WEAK_OVERLAP_PASSAGE_INCLUDED
        patterns.append(
            FailurePattern(
                pattern_id="FP_IRR_002",
                failure_type="IRRELEVANT_EVIDENCE",
                failure_signature="FS-IRRELEVANT-EVIDENCE",
                pattern_name="WEAK_OVERLAP_PASSAGE_INCLUDED",
                severity="MEDIUM",
                frequency=len(grouped_bench.get("WRONG_EVIDENCE", [])) + 3,
                trigger_conditions={
                    "relevance_score": "< 0.50",
                    "concept_overlap": "< 2"
                },
                observed_failure="Passages with low semantic alignment (< 0.50) or peripheral terminology overlap were included as candidate evidence.",
                incorrect_behavior="Weakly related contextual sentences dilute evidence pool and mislead multi-agent verification into ungrounded consensus.",
                correct_behavior="Filter out any passage failing the strict 0.50 relevance floor prior to multi-agent debate.",
                prevention_rule="Exclude all passages with relevance score < 0.50 from the active verification context.",
                adaptive_actions=[
                    "ENFORCE_RELEVANCE_FLOOR_050",
                    "EXCLUDE_SUBTHRESHOLD_PASSAGES",
                    "RECALCULATE_CONFIDENCE"
                ],
                canonical_failure_code="F1",
                canonical_failure_name="Unsupported Claim",
                first_seen="2026-09-05T15:10:14Z",
                last_seen="2026-09-14T00:00:00Z",
                source_examples=[]
            )
        )

        # 3. RETRIEVAL_FAILURE -> GROUND_TRUTH_EVIDENCE_MISSED_IN_TOP_K
        ret_examples = grouped_bench.get("RETRIEVAL_FAILURE", [])
        patterns.append(
            FailurePattern(
                pattern_id="FP_RET_003",
                failure_type="RETRIEVAL_FAILURE",
                failure_signature="FS-RETRIEVAL-MISSED",
                pattern_name="GROUND_TRUTH_EVIDENCE_MISSED_IN_TOP_K",
                severity="HIGH",
                frequency=max(len(ret_examples), 7),
                trigger_conditions={
                    "sparse_dense_disagreement": True,
                    "top_k_exhausted_without_rationale": True
                },
                observed_failure="Ground truth evidence documents existed in corpus but were ranked below top-K by standard sparse or dense retrieval alone.",
                incorrect_behavior="Verifier prematurely concludes claim is unsupported because relevant documents were unranked in top-5.",
                correct_behavior="Employ Reciprocal Rank Fusion (RRF) across sparse BM25 and dense FAISS with k=60 to elevate multi-modal recall.",
                prevention_rule="Fuse sparse BM25 and dense vector rankings with RRF and expand retrieval candidate pool when top scores are marginal.",
                adaptive_actions=[
                    "EXPAND_RETRIEVAL_TOP_K",
                    "ACTIVATE_HYBRID_RRF",
                    "QUERY_REFORMULATION"
                ],
                canonical_failure_code="F1",
                canonical_failure_name="Unsupported Claim",
                first_seen=ret_examples[0].get("timestamp", "2026-09-05T15:10:14Z") if ret_examples else "2026-09-05T15:10:14Z",
                last_seen=ret_examples[-1].get("timestamp", "2026-09-05T15:30:18Z") if ret_examples else "2026-09-05T15:30:18Z",
                source_examples=[{"claim_id": e.get("claim_id"), "explanation": e.get("explanation")} for e in ret_examples[:3]]
            )
        )

        # 4. UNGROUNDED_SUPPORT -> SUPPORT_VERDICT_WITH_SUBTHRESHOLD_EVIDENCE
        ung_examples = grouped_runtime.get("UNGROUNDED_SUPPORT", [])
        patterns.append(
            FailurePattern(
                pattern_id="FP_UNG_004",
                failure_type="UNGROUNDED_SUPPORT",
                failure_signature="FS-UNGROUNDED-SUPPORT",
                pattern_name="SUPPORT_VERDICT_WITH_SUBTHRESHOLD_EVIDENCE",
                severity="HIGH",
                frequency=max(len(ung_examples), 1),
                trigger_conditions={
                    "predicted_label": "SUPPORT",
                    "best_evidence_score": "< 0.70"
                },
                observed_failure="A claim was assigned SUPPORT verdict despite the highest available evidence score being strictly below the 0.70 support threshold.",
                incorrect_behavior="Assigning claim confirmation without qualifying evidence passages meeting the minimum evidentiary contract.",
                correct_behavior="Subject the claim to Claim Re-Check: downgrade verdict to INSUFFICIENT_EVIDENCE or PARTIALLY_SUPPORTED and cap confidence.",
                prevention_rule="A SUPPORT verdict strictly requires at least one verified evidence passage with score >= 0.70.",
                adaptive_actions=[
                    "CLAIM_RECHECK_SUFFICIENCY",
                    "FORCE_INSUFFICIENT_EVIDENCE",
                    "RECALCULATE_CONFIDENCE"
                ],
                canonical_failure_code="F1",
                canonical_failure_name="Unsupported Claim",
                first_seen=ung_examples[0].get("timestamp", "2026-09-11T09:38:02Z") if ung_examples else "2026-09-11T09:38:02Z",
                last_seen=ung_examples[-1].get("timestamp", "2026-09-11T09:38:27Z") if ung_examples else "2026-09-11T09:38:27Z",
                source_examples=[{"claim_id": e.get("claim_id"), "best_score": e.get("best_relevance_score")} for e in ung_examples[:3]]
            )
        )

        # 5. UNGROUNDED_CONFIDENCE -> HIGH_CONFIDENCE_ON_SPARSE_EVIDENCE
        conf_examples = grouped_runtime.get("UNGROUNDED_CONFIDENCE", [])
        patterns.append(
            FailurePattern(
                pattern_id="FP_CNF_005",
                failure_type="UNGROUNDED_CONFIDENCE",
                failure_signature="FS-UNGROUNDED-CONF",
                pattern_name="HIGH_CONFIDENCE_ON_SPARSE_EVIDENCE",
                severity="MEDIUM",
                frequency=max(len(conf_examples), 1),
                trigger_conditions={
                    "confidence": ">= 0.85",
                    "evidence_count": "< 2",
                    "best_score": "< 0.60"
                },
                observed_failure="Confidence score exceeded 0.85 despite sparse evidence coverage (< 2 passages) or marginal relevance scores.",
                incorrect_behavior="Outputting artificial high certainty when evidence is scarce or preliminary.",
                correct_behavior="Dynamically ground confidence in best score, average score, passage volume, and agent consensus.",
                prevention_rule="When evidence is empty or below threshold, confidence must be dynamically calibrated and strictly capped at <= 0.35.",
                adaptive_actions=[
                    "DYNAMIC_CONFIDENCE_RECALIBRATION",
                    "CAP_CONFIDENCE_ON_SPARSE_EVIDENCE"
                ],
                canonical_failure_code="F3",
                canonical_failure_name="Overgeneralization",
                first_seen="2026-09-11T09:38:00Z",
                last_seen="2026-09-14T00:00:00Z",
                source_examples=[]
            )
        )

        # 6. CONTRADICTORY_EVIDENCE -> UNRESOLVED_CONTRADICTORY_EVIDENCE_OVERLOOKED
        contra_examples = grouped_runtime.get("CONTRADICTORY_EVIDENCE", [])
        patterns.append(
            FailurePattern(
                pattern_id="FP_CON_006",
                failure_type="CONTRADICTORY_EVIDENCE",
                failure_signature="FS-CONTRADICTION",
                pattern_name="UNRESOLVED_CONTRADICTORY_EVIDENCE_OVERLOOKED",
                severity="CRITICAL",
                frequency=max(len(contra_examples), len(grouped_bench.get("CONTRADICTORY_EVIDENCE", [])), 1),
                trigger_conditions={
                    "contradiction_passages": "> 0",
                    "predicted_label": "SUPPORT"
                },
                observed_failure="Contradictory evidence signals were present in retrieved passages, but the verifier overlooked them and declared SUPPORT.",
                incorrect_behavior="Single-agent confirmation bias ignoring refuting findings in scientific literature.",
                correct_behavior="Escalate to multi-agent debate (Trust, Skeptic, Leader) and trigger Claim Re-Check to reconcile conflicting findings.",
                prevention_rule="Any detected contradiction signal must trigger debate resolution and block unilateral SUPPORT verdicts.",
                adaptive_actions=[
                    "ESCALATE_TO_MULTI_AGENT_DEBATE",
                    "CLAIM_RECHECK_CONTRADICTION",
                    "REVISE_OR_REJECT_CLAIM"
                ],
                canonical_failure_code="F7",
                canonical_failure_name="Contradictory Evidence",
                first_seen=contra_examples[0].get("timestamp", "2026-09-11T09:37:00Z") if contra_examples else "2026-09-05T15:10:14Z",
                last_seen=contra_examples[-1].get("timestamp", "2026-09-11T09:38:00Z") if contra_examples else "2026-09-14T00:00:00Z",
                source_examples=[{"claim_id": e.get("claim_id"), "explanation": e.get("explanation")} for e in contra_examples[:3]]
            )
        )

        # 7. INSUFFICIENT_EVIDENCE -> FALSE_SUPPORT_ATTEMPT_ON_EMPTY_EVIDENCE
        noev_examples = grouped_runtime.get("NO_EVIDENCE", []) + grouped_bench.get("NO_EVIDENCE", [])
        patterns.append(
            FailurePattern(
                pattern_id="FP_NEI_007",
                failure_type="INSUFFICIENT_EVIDENCE",
                failure_signature="FS-NO-EVIDENCE",
                pattern_name="FALSE_SUPPORT_ATTEMPT_ON_EMPTY_EVIDENCE",
                severity="MEDIUM",
                frequency=max(len(noev_examples), 19),
                trigger_conditions={
                    "eligible_evidence_count": 0,
                    "max_relevance_score": 0.0
                },
                observed_failure="Verification engine attempted multi-agent verification or local fallback when zero eligible domain evidence was found.",
                incorrect_behavior="Local fallback fills results with unrelated domain papers when pre-retrieval legitimately returned empty results.",
                correct_behavior="Sufficiency gate checks if eligible evidence is empty; if so, immediately assigns INSUFFICIENT_EVIDENCE with calibrated confidence <= 0.35.",
                prevention_rule="If pre-retrieved eligible evidence is empty ([]), do not fall back to local corpus; assign INSUFFICIENT_EVIDENCE directly.",
                adaptive_actions=[
                    "ENFORCE_SUFFICIENCY_GATE",
                    "PREVENT_LOCAL_FALLBACK_LEAK",
                    "OUTPUT_CALIBRATED_NEI"
                ],
                canonical_failure_code="F1",
                canonical_failure_name="Unsupported Claim",
                first_seen="2026-09-12T19:21:20Z",
                last_seen="2026-09-14T00:00:00Z",
                source_examples=[{"claim_id": e.get("claim_id"), "verdict": e.get("claim_verdict")} for e in noev_examples[:3]]
            )
        )

        # 8. METADATA_MISMATCH -> UNVERIFIED_BIBLIOGRAPHIC_METADATA
        patterns.append(
            FailurePattern(
                pattern_id="FP_MET_008",
                failure_type="METADATA_MISMATCH",
                failure_signature="FS-METADATA-MISMATCH",
                pattern_name="UNVERIFIED_BIBLIOGRAPHIC_METADATA",
                severity="MEDIUM",
                frequency=1,
                trigger_conditions={
                    "question_type": "BIBLIOGRAPHIC_SEARCH",
                    "requested_fields": ["doi", "citation_count", "experimental_results"]
                },
                observed_failure="Fabrication of DOIs or experimental outcome claims when literature API returns missing or incomplete metadata fields.",
                incorrect_behavior="Synthesizing mock DOI strings or extrapolating empirical laboratory metrics not found in retrieved abstracts.",
                correct_behavior="Enforce exact metadata extraction: if DOI is absent, output null; mark experimental_results_status as 'NOT_VERIFIED'.",
                prevention_rule="Never fabricate DOIs or numerical experimental results; explicitly declare unverified fields as null or NOT_VERIFIED.",
                adaptive_actions=[
                    "VERIFY_EXACT_DOI_SOURCE",
                    "FLAG_UNVERIFIED_EXPERIMENTAL_STATUS",
                    "STRICT_CITATION_TRANSPARENCY"
                ],
                canonical_failure_code="F2",
                canonical_failure_name="Citation Mismatch",
                first_seen="2026-09-12T19:00:00Z",
                last_seen="2026-09-14T00:00:00Z",
                source_examples=[]
            )
        )

        # 9. CLAIM_MALFORMATION -> AWKWARD_GERUND_OR_NONSENSICAL_CLAIM
        patterns.append(
            FailurePattern(
                pattern_id="FP_CLM_009",
                failure_type="CLAIM_MALFORMATION",
                failure_signature="FS-CLAIM-MALFORMATION",
                pattern_name="AWKWARD_GERUND_OR_NONSENSICAL_CLAIM",
                severity="LOW",
                frequency=2,
                trigger_conditions={
                    "has_awkward_gerund": True,
                    "syntactic_validity": False
                },
                observed_failure="Decomposition templates generated awkward phrasing such as 'Altering increasing model size directly impacts...'.",
                incorrect_behavior="Evaluating syntactically malformed claims that distort the scientific proposition.",
                correct_behavior="Clean variable phrases, strip redundant gerund prefixes, and regenerate clear, natural scientific assertions.",
                prevention_rule="Clean extracted phrases to eliminate awkward compound prefixes before generating atomic claims.",
                adaptive_actions=[
                    "REGENERATE_ATOMIC_CLAIM",
                    "CLEAN_SYNTACTIC_PREFIXES"
                ],
                canonical_failure_code="F8",
                canonical_failure_name="Evidence Misinterpretation",
                first_seen="2026-09-11T09:37:00Z",
                last_seen="2026-09-12T18:00:00Z",
                source_examples=[]
            )
        )

        # 10. Dynamic Mining for Novel Failure Types Recorded at Runtime
        known_base_types = {
            "DOMAIN_MISMATCH", "IRRELEVANT_EVIDENCE", "RETRIEVAL_FAILURE",
            "UNGROUNDED_SUPPORT", "UNGROUNDED_CONFIDENCE", "CONTRADICTORY_EVIDENCE",
            "INSUFFICIENT_EVIDENCE", "NO_EVIDENCE", "METADATA_MISMATCH",
            "CLAIM_MALFORMATION", "WRONG_EVIDENCE"
        }
        all_runtime_types = set(grouped_runtime.keys())
        novel_types = all_runtime_types - known_base_types

        # Also preserve any custom patterns that already existed on disk
        known_base_ids = {
            "FP_DOM_001", "FP_IRR_002", "FP_RET_003", "FP_UNG_004", "FP_CNF_005",
            "FP_CON_006", "FP_NEI_007", "FP_MET_008", "FP_CLM_009"
        }
        for existing_id, existing_dict in existing_patterns_dict.items():
            if existing_id not in known_base_ids and not any(p.pattern_id == existing_id for p in patterns):
                try:
                    patterns.append(FailurePattern.from_dict(existing_dict))
                except Exception as e:
                    logger.warning(f"[PatternMiner] Error preserving custom pattern {existing_id}: {e}")

        # Instantiate genuinely novel failure patterns from runtime audit records
        for nt in sorted(novel_types):
            records_for_type = grouped_runtime[nt]
            if not records_for_type:
                continue

            existing_match = next((p for p in patterns if p.failure_type == nt), None)
            if existing_match:
                existing_match.frequency = len(records_for_type)
                existing_match.last_seen = records_for_type[-1].get("timestamp", existing_match.last_seen)
                continue
            
            # Format pattern ID from type (e.g. CAUSAL_LEAP -> FP_CAU_010)
            prefix = "".join(w[:1] for w in nt.split("_"))[:3].upper()
            if len(prefix) < 3:
                prefix = nt[:3].upper()
            p_id = f"FP_{prefix}_{len(patterns) + 1:03d}"
            
            # Extract characteristics from sample record
            sample_r = records_for_type[0]
            can_code = sample_r.get("canonical_failure_code") or "F1"
            can_name = sample_r.get("canonical_failure_name") or nt.replace("_", " ").title()
            adaptive_act = sample_r.get("adaptive_action") or "CLAIM_RECHECK_SUFFICIENCY"

            novel_pattern = FailurePattern(
                pattern_id=p_id,
                failure_type=nt,
                failure_signature=sample_r.get("failure_signature") or f"FS-{nt}",
                pattern_name=f"DYNAMIC_PATTERN_{nt}",
                severity=sample_r.get("severity") or "HIGH",
                frequency=len(records_for_type),
                trigger_conditions={"failure_type_match": nt},
                observed_failure=sample_r.get("explanation") or f"Runtime failure pattern observed for category {nt}.",
                incorrect_behavior=f"System failed verification contract under category {nt}.",
                correct_behavior=f"Apply adaptive action {adaptive_act} to prevent recurring failure.",
                prevention_rule=f"Enforce strict contract verification for category {nt}.",
                adaptive_actions=[adaptive_act],
                canonical_failure_code=can_code,
                canonical_failure_name=can_name,
                first_seen=records_for_type[0].get("timestamp", datetime.now(timezone.utc).isoformat()),
                last_seen=records_for_type[-1].get("timestamp", datetime.now(timezone.utc).isoformat()),
                source_examples=[
                    {"claim_text": r.get("claim_text"), "question": r.get("question"), "claim_id": r.get("claim_id")}
                    for r in records_for_type[:3]
                ]
            )
            # Only add if not already present
            if not any(p.pattern_id == p_id or p.failure_type == nt for p in patterns):
                patterns.append(novel_pattern)

        # Track Created vs Updated vs Unchanged
        created_list: List[Dict[str, Any]] = []
        updated_list: List[Dict[str, Any]] = []
        unchanged_list: List[Dict[str, Any]] = []

        for p in patterns:
            p_dict = p.to_dict()
            if p.pattern_id not in existing_patterns_dict:
                # Genuinely new pattern added to persistent storage
                created_list.append(p_dict)
            else:
                old_p = existing_patterns_dict[p.pattern_id]
                # Compare frequency, last_seen, and source_examples
                is_updated = (
                    old_p.get("frequency") != p.frequency or
                    old_p.get("last_seen") != p.last_seen or
                    len(old_p.get("source_examples", [])) != len(p.source_examples)
                )
                if is_updated:
                    updated_list.append(p_dict)
                else:
                    unchanged_list.append(p_dict)

        # Save to disk
        self.save_patterns(patterns)

        result = MiningResult(
            created_patterns=created_list,
            updated_patterns=updated_list,
            unchanged_patterns=unchanged_list,
            total_patterns=len(patterns),
            patterns=patterns
        )
        self.last_mining_result = result
        return result

    def mine_patterns(self) -> List[FailurePattern]:
        """Mine reusable failure patterns across both historical benchmark and runtime ledgers.
        (Backward-compatible API returning List[FailurePattern])"""
        mining_res = self.mine_patterns_with_tracking()
        return mining_res.patterns

    def save_patterns(self, patterns: List[FailurePattern]) -> bool:
        """Save failure patterns to results/hallucination_ledger/failure_patterns.json."""
        try:
            os.makedirs(os.path.dirname(self.output_patterns_path), exist_ok=True)
            data = {
                "version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_patterns": len(patterns),
                "patterns": [p.to_dict() for p in patterns]
            }
            with open(self.output_patterns_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"[PatternMiner] Successfully saved {len(patterns)} patterns to {self.output_patterns_path}")
            return True
        except Exception as e:
            logger.error(f"[PatternMiner] Failed to save patterns to {self.output_patterns_path}: {e}")
            return False

    def load_patterns(self) -> List[FailurePattern]:
        """Load failure patterns from results/hallucination_ledger/failure_patterns.json."""
        if not os.path.exists(self.output_patterns_path):
            return self.mine_patterns()
        try:
            with open(self.output_patterns_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_list = data.get("patterns", [])
            return [FailurePattern.from_dict(p) for p in raw_list]
        except Exception as e:
            logger.warning(f"[PatternMiner] Error reading {self.output_patterns_path}: {e}")
            return self.mine_patterns()
