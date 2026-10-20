"""Failure Memory Engine for CiteGuard.

Implements the complete CiteGuard closed loop:
1. Memory Gate & Trusted Memory: Stores cleanly passed verifications.
2. Historical Failure Matching: Evaluates claim + candidate evidence against learned failure patterns.
3. Risk Assessment: Quantifies risk level (LOW, MEDIUM, HIGH, VERY_HIGH).
4. Adaptive Verification Coordinator: Dictates verification depth, additional retrieval, debate escalation,
   and contradiction checking.
5. Claim Re-Check: Strictly evaluates evidence under heightened scrutiny, producing canonical actions:
   ACCEPT / REVISE / REJECT.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from src.evaluation.failure_pattern import FailurePattern, PatternMatchResult
from src.evaluation.pattern_miner import PatternMiner, DEFAULT_PATTERNS_FILE

logger = logging.getLogger(__name__)

DEFAULT_TRUSTED_MEMORY_PATH = os.path.join("results", "hallucination_ledger", "trusted_memory.json")


class FailureMemory:
    """Persistent Failure-Pattern Memory and Adaptive Verification Coordinator."""

    def __init__(
        self,
        patterns_path: str = DEFAULT_PATTERNS_FILE,
        trusted_memory_path: str = DEFAULT_TRUSTED_MEMORY_PATH,
        miner: Optional[PatternMiner] = None
    ):
        self.patterns_path = patterns_path
        self.trusted_memory_path = trusted_memory_path
        self.miner = miner or PatternMiner(output_patterns_path=self.patterns_path)
        self.patterns: List[FailurePattern] = self.miner.load_patterns()
        os.makedirs(os.path.dirname(self.trusted_memory_path), exist_ok=True)

    def reload_patterns(self) -> int:
        """Reload failure patterns from disk."""
        self.patterns = self.miner.load_patterns()
        return len(self.patterns)

    def mine_patterns_from_historical_ledger(self) -> List[FailurePattern]:
        """Mine and persist failure patterns from historical benchmark and runtime ledgers."""
        mining_res = self.miner.mine_patterns_with_tracking()
        self.patterns = mining_res.patterns
        self.last_mining_result = mining_res
        return self.patterns

    def get_last_mining_result(self) -> Optional[Any]:
        """Return the structured MiningResult from the most recent mining operation."""
        res = getattr(self.miner, "last_mining_result", None)
        if res is None and self.miner is not None:
            try:
                self.mine_patterns_from_historical_ledger()
                res = getattr(self.miner, "last_mining_result", None)
            except Exception as e:
                logger.warning(f"Failed to compute initial mining result: {e}")
        return res

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Return memory statistics regarding stored patterns and trusted memory claims."""
        trusted_claims = self.load_trusted_memory()
        return {
            "total_patterns": len(self.patterns),
            "trusted_memory_records": len(trusted_claims),
            "patterns_path": self.patterns_path,
            "trusted_memory_path": self.trusted_memory_path
        }

    # ----------------------------------------------------------------------
    # 1. MEMORY GATE & TRUSTED MEMORY
    # ----------------------------------------------------------------------

    def is_eligible_for_trusted_memory(
        self,
        verdict: str,
        confidence: float,
        evidence: List[Any],
        pattern_matched: bool = False,
        recheck_status: str = "ACCEPT",
        allow_partial: bool = False
    ) -> Tuple[bool, str]:
        """Evaluates whether a verification result satisfies the Trusted Memory policy.
        
        Policy Rules:
        1. Fully supported: verdict must normalize to SUPPORT / SUPPORTED.
        2. Unsupported, contradictory, or ungrounded claims are rejected.
        3. Partially supported claims are rejected by default (unless allow_partial=True).
        4. Confidence must be >= 0.70.
        5. Genuine evidence required (evidence_count >= 1).
        6. Claims flagged by active failure patterns or rejected by adaptive re-check cannot be stored.
        """
        v_norm = str(verdict or "").upper().replace(" ", "_")
        if "." in v_norm:
            v_norm = v_norm.split(".")[-1]

        if pattern_matched:
            return False, "Excluded: Matched active failure pattern (unresolved risk)"

        if str(recheck_status).upper() in ("REVISE", "REJECT"):
            return False, f"Excluded: Adaptive re-check status is {recheck_status}"

        if v_norm in ("CONTRADICT", "CONTRADICTED", "REFUTED"):
            return False, "Excluded: Contradicted claim cannot be admitted to Trusted Memory"

        if v_norm in ("INSUFFICIENT_EVIDENCE", "NEI", "NO_EVIDENCE", "UNSUPPORTED", "CLEAN"):
            return False, f"Excluded: Insufficient or missing evidence (verdict: {v_norm})"

        if v_norm in ("PARTIALLY_SUPPORTED", "PARTIAL"):
            if not allow_partial:
                return False, "Excluded: Policy restricts Trusted Memory to fully SUPPORTED claims (partial evidence requires further substantiation)"

        if v_norm not in ("SUPPORT", "SUPPORTED") and not (allow_partial and v_norm in ("PARTIALLY_SUPPORTED", "PARTIAL")):
            return False, f"Excluded: Verdict {v_norm} does not qualify as supported"

        if float(confidence or 0.0) < 0.70:
            return False, f"Excluded: Confidence {confidence:.2f} is below trust threshold 0.70"

        if not evidence or len(evidence) == 0:
            return False, "Excluded: Zero evidence passages attached"

        return True, "Eligible: Satisfies all Trusted Memory acceptance criteria"

    def record_trusted_verification(
        self,
        claim_id: str,
        claim_text: str,
        verdict: str,
        confidence: float,
        evidence: List[Any],
        question: str,
        pattern_matched: bool = False,
        recheck_status: str = "ACCEPT",
        allow_partial: bool = False
    ) -> bool:
        """Pass branch: Register cleanly verified claim into Trusted Memory following strict trust policy."""
        eligible, reason = self.is_eligible_for_trusted_memory(
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            pattern_matched=pattern_matched,
            recheck_status=recheck_status,
            allow_partial=allow_partial
        )
        if not eligible:
            logger.info(f"[FailureMemory] Claim {claim_id} rejected from Trusted Memory: {reason}")
            return False

        try:
            trusted_records = self.load_trusted_memory()
            norm_text = claim_text.strip().lower()

            # Deduplication & Versioning Policy:
            # Check for existing entry with same claim_id or identical normalized claim_text
            existing_idx = None
            for idx, r in enumerate(trusted_records):
                if r.get("claim_id") == claim_id or r.get("claim_text", "").strip().lower() == norm_text:
                    existing_idx = idx
                    break

            rec = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_count": len(evidence),
                "question": question,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if existing_idx is not None:
                # Deduplication: Update existing record with latest verification
                trusted_records[existing_idx] = rec
            else:
                trusted_records.append(rec)

            with open(self.trusted_memory_path, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "total_trusted_claims": len(trusted_records),
                    "claims": trusted_records
                }, f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"[FailureMemory] Failed to record to trusted memory: {e}")
            return False

    def load_trusted_memory(self) -> List[Dict[str, Any]]:
        """Load records from trusted memory."""
        if not os.path.exists(self.trusted_memory_path):
            return []
        try:
            with open(self.trusted_memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("claims", [])
        except Exception as e:
            logger.warning(f"[FailureMemory] Failed reading trusted memory: {e}")
            return []

    # ----------------------------------------------------------------------
    # 2. HISTORICAL FAILURE MATCHING & RISK ASSESSMENT
    # ----------------------------------------------------------------------

    def match_historical_failures(
        self,
        claim_text: str,
        evidence: List[Any],
        question: Optional[str] = None,
        preliminary_verdict: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> PatternMatchResult:
        """Match claim + evidence configuration against learned failure patterns.
        
        Calculates similarity/risk score and assigns an adaptive verification policy.
        """
        if not self.patterns:
            self.reload_patterns()

        q_lower = (question or "").lower()
        c_lower = claim_text.lower()
        pred_norm = str(preliminary_verdict or "SUPPORT").upper()
        conf_val = float(confidence or 0.85)

        matched_patterns: List[FailurePattern] = []
        reasons: List[str] = []
        max_score = 0.0

        # Helper to extract evidence text and metadata
        ev_scores = []
        ev_texts = []
        ev_eligible = True
        has_domain_mismatch = False
        contra_count = 0

        for ev in evidence:
            sc = float(getattr(ev, "score", 0.0) or (ev.get("score") if isinstance(ev, dict) else 0.0) or 0.0)
            t = str(getattr(ev, "text", "") or (ev.get("text") if isinstance(ev, dict) else "") or "").lower()
            meta = getattr(ev, "metadata", {}) if hasattr(ev, "metadata") else (ev.get("metadata", {}) if isinstance(ev, dict) else {})
            rel = str(getattr(ev, "relation", "") or (ev.get("relation") if isinstance(ev, dict) else "") or "").upper()
            rel_label = str(meta.get("relationship", "")).upper()

            ev_scores.append(sc)
            ev_texts.append(t)

            if meta.get("evidence_eligible") is False or meta.get("topic_match") is False or "DOMAIN_MISMATCH" in rel:
                has_domain_mismatch = True
            if "CONTRADICT" in rel or "CONTRADICT" in rel_label or "REFUTE" in rel:
                contra_count += 1

        best_score = max(ev_scores, default=0.0)

        # Check for outdated evidence signals
        has_outdated = False
        for ev in evidence:
            meta = getattr(ev, "metadata", {}) if hasattr(ev, "metadata") else (ev.get("metadata", {}) if isinstance(ev, dict) else {})
            pub_year = meta.get("year") or meta.get("publication_year") or getattr(ev, "year", None)
            if meta.get("is_superseded") is True or meta.get("outdated") is True or meta.get("is_outdated") is True or meta.get("superseded") is True:
                has_outdated = True
            elif isinstance(pub_year, int) and pub_year > 0:
                if 2026 - pub_year > 25:
                    has_outdated = True
                elif 2026 - pub_year > 15 and ("quantum" in q_lower or "llm" in q_lower or "transformer" in q_lower or "2026" in c_lower):
                    has_outdated = True

        # Domain mismatch checks: Quantum / FL claim with unrelated topic (smart grid, power grid, etc.)
        is_quantum_fl = ("quantum" in q_lower or "quantum" in c_lower) and ("federated" in q_lower or "federated" in c_lower)
        unrelated_terms = ["smart grid", "power grid", "electric grid", "drug discovery", "natural products", "carcinoma", "tumor"]
        has_unrelated_text = any(any(u in t for u in unrelated_terms) for t in ev_texts)

        # Check against patterns:
        for p in self.patterns:
            p_id = p.pattern_id

            # FP_DOM_001: DOMAIN_MISMATCH
            if p_id == "FP_DOM_001":
                if (is_quantum_fl and has_unrelated_text) or has_domain_mismatch:
                    matched_patterns.append(p)
                    reasons.append("Claim relates to quantum-resistant federated learning but candidate evidence contains off-domain keywords (e.g. smart grids or drug discovery).")
                    max_score = max(max_score, 0.92)

            # FP_CON_006: CONTRADICTORY_EVIDENCE
            elif p_id == "FP_CON_006":
                if contra_count > 0 and pred_norm in ("SUPPORT", "SUPPORTED"):
                    matched_patterns.append(p)
                    reasons.append(f"Detected {contra_count} contradictory evidence passages while preliminary verdict is {pred_norm}.")
                    max_score = max(max_score, 0.95)

            # FP_UNG_004: UNGROUNDED_SUPPORT
            elif p_id == "FP_UNG_004":
                if pred_norm in ("SUPPORT", "SUPPORTED") and (best_score < 0.70 or len(evidence) == 0):
                    matched_patterns.append(p)
                    reasons.append(f"Preliminary verdict is SUPPORT but highest evidence score is {best_score:.2f} (< 0.70 support threshold).")
                    max_score = max(max_score, 0.88)

            # FP_CNF_005: UNGROUNDED_CONFIDENCE
            elif p_id == "FP_CNF_005":
                if conf_val >= 0.85 and (len(evidence) < 2 and best_score < 0.60):
                    matched_patterns.append(p)
                    reasons.append(f"High confidence ({conf_val*100:.0f}%) assigned on sparse or weak evidence coverage.")
                    max_score = max(max_score, 0.80)

            # FP_NEI_007: INSUFFICIENT_EVIDENCE
            elif p_id == "FP_NEI_007":
                if len(evidence) == 0 or (has_domain_mismatch and best_score <= 0.0):
                    matched_patterns.append(p)
                    reasons.append("Zero eligible domain evidence passages available for claim.")
                    max_score = max(max_score, 0.90)

            # FP_CLM_009: CLAIM_MALFORMATION / MISINTERPRETATION (F8)
            elif p_id == "FP_CLM_009":
                has_misinterpretation = any(
                    (meta.get("misinterpreted") is True or meta.get("directional_error") is True)
                    for meta in [getattr(e, "metadata", {}) if hasattr(e, "metadata") else (e.get("metadata", {}) if isinstance(e, dict) else {}) for e in evidence]
                )
                if "altering increasing" in c_lower or "directly affects target scientific outcomes" in c_lower or has_misinterpretation:
                    matched_patterns.append(p)
                    reasons.append("Claim phrasing exhibits awkward syntax or retrieved evidence misinterpretation / directional error.")
                    max_score = max(max_score, 0.85)

            # FP_MET_008: Citation Mismatch (F2)
            elif p_id == "FP_MET_008":
                has_citation_mismatch = any(
                    (meta.get("citation_mismatch") is True or meta.get("metadata_unverified") is True)
                    for meta in [getattr(e, "metadata", {}) if hasattr(e, "metadata") else (e.get("metadata", {}) if isinstance(e, dict) else {}) for e in evidence]
                )
                if has_citation_mismatch:
                    matched_patterns.append(p)
                    reasons.append("Cited paper, DOI, or bibliographic provenance cannot be verified in the scientific registry.")
                    max_score = max(max_score, 0.90)

            # Generalized / Dynamically Learned Failure Pattern Matching:
            else:
                trig = p.trigger_conditions or {}
                matched_dyn = False
                dyn_reason = ""
                cf_code = p.canonical_failure_code
                ft_type = p.failure_type

                # F6 / Context Mismatch / Population Mismatch
                if cf_code == "F6" or ft_type in ("DOMAIN_MISMATCH", "POPULATION_MISMATCH", "CONTEXT_MISMATCH", "CROSS_SPECIES_EXTRAPOLATION"):
                    human_markers = ["in human", "human clinical", "cancer patients", "clinical patients", "humans", "patients", "human subjects"]
                    preclin_markers = ["in vitro", "cell culture", "cell lines", "u87", "petri", "murine model", "mouse model", "rat model", "mice"]
                    if (is_quantum_fl and has_unrelated_text) or has_domain_mismatch:
                        matched_dyn = True
                        dyn_reason = "Evidence context mismatch: Off-domain or incompatible context."
                    elif any(hm in c_lower for hm in human_markers) and any(any(pm in t for pm in preclin_markers) for t in ev_texts):
                        matched_dyn = True
                        dyn_reason = "Population/Context Mismatch: Preclinical in vitro/animal evidence used for human clinical assertion."

                # F4 / Causal Leap
                elif cf_code == "F4" or ft_type in ("CAUSAL_LEAP", "REASONING_FAILURE", "SURROGATE_ENDPOINT_LEAP"):
                    causal_markers_local = ["causes", "causally triggers", "directly proves", "is the cause of", "leads directly to"]
                    correlational_markers_local = ["correlated with", "associated with", "observational cohort", "co-occurs with", "trend toward"]
                    if any(cm in c_lower for cm in causal_markers_local) and any(any(cem in t for cem in correlational_markers_local) for t in ev_texts):
                        matched_dyn = True
                        dyn_reason = "Causal leap: Direct causality asserted on observational correlation."

                # F5 / Outdated Evidence / Source Quality
                elif cf_code == "F5" or ft_type in ("OUTDATED_EVIDENCE", "SOURCE_QUALITY_FAILURE", "OBSOLETE_EVIDENCE"):
                    if has_outdated:
                        matched_dyn = True
                        dyn_reason = "Chronologically obsolete or superseded evidence detected."

                # F3 / Overgeneralization
                elif cf_code == "F3" or ft_type in ("OVERGENERALIZATION", "UNGROUNDED_CONFIDENCE"):
                    overgen_quantifiers_local = ["all human", "across all", "all populations", "worldwide", "permanently", "every human", "universally"]
                    restricted_cohort_local = ["24 healthy", "trial of 24", "12 mice", "mice", "mouse", "rat", "in vitro", "small cohort", "college students"]
                    if any(oq in c_lower for oq in overgen_quantifiers_local) and any(any(rc in t for rc in restricted_cohort_local) for t in ev_texts):
                        matched_dyn = True
                        dyn_reason = "Overgeneralization: Broad universal claim extrapolated from narrow sample."
                    elif conf_val >= 0.85 and (len(evidence) < 2 and best_score < 0.60):
                        matched_dyn = True
                        dyn_reason = "High confidence on sparse/narrow evidence."

                # F7 / Contradictory Evidence
                elif cf_code == "F7" or ft_type == "CONTRADICTORY_EVIDENCE":
                    if contra_count > 0 and pred_norm in ("SUPPORT", "SUPPORTED"):
                        matched_dyn = True
                        dyn_reason = f"Contradictory evidence detected ({contra_count} refuting passages)."

                # F1 / Unsupported Claim
                elif cf_code == "F1" or ft_type in ("UNSUPPORTED_CLAIM", "NO_EVIDENCE", "INSUFFICIENT_EVIDENCE", "IRRELEVANT_EVIDENCE"):
                    if len(evidence) == 0 or best_score < 0.50:
                        matched_dyn = True
                        dyn_reason = "Insufficient or ungrounded evidence."

                # Direct trigger conditions match:
                if not matched_dyn and trig:
                    if trig.get("failure_type_match") and trig.get("failure_type_match") == ft_type:
                        if trig.get("is_superseded") and has_outdated:
                            matched_dyn = True
                            dyn_reason = f"Learned trigger match: superseded evidence ({p.pattern_id})."
                        elif trig.get("target_domains_conflict") and has_domain_mismatch:
                            matched_dyn = True
                            dyn_reason = f"Learned trigger match: domain conflict ({p.pattern_id})."

                if matched_dyn:
                    matched_patterns.append(p)
                    reasons.append(f"Matched learned persistent pattern {p.pattern_id} ({p.pattern_name}): {dyn_reason or p.observed_failure}")
                    max_score = max(max_score, 0.86)

        # F4: Causal Leap fallback if not already in persistent patterns
        causal_claim_markers = ["causes", "causally triggers", "directly proves", "is the cause of", "leads directly to"]
        correlational_markers = ["correlated with", "associated with", "observational cohort", "co-occurs with", "trend toward"]
        if not any(p.canonical_failure_code == "F4" for p in matched_patterns):
            if any(cm in c_lower for cm in causal_claim_markers) and any(any(cem in t for cem in correlational_markers) for t in ev_texts):
                fp_cau = FailurePattern(
                    pattern_id="FP_CAU_010",
                    failure_type="CAUSAL_LEAP",
                    failure_signature="FS-CAUSAL-LEAP",
                    pattern_name="CAUSAL_CLAIM_ON_OBSERVATIONAL_DATA",
                    severity="HIGH",
                    frequency=5,
                    trigger_conditions={"causal_claim": True, "observational_evidence": True},
                    observed_failure="Claim asserts direct causality while evidence only substantiates observational correlation.",
                    incorrect_behavior="Confirming causal relationship from non-interventional or correlational literature.",
                    correct_behavior="Qualify claim to observational association or revise verdict to PARTIALLY_SUPPORTED.",
                    prevention_rule="Require interventional randomized trial or qualify causal verbs to associative terminology.",
                    adaptive_actions=["QUALIFY_CAUSAL_CLAIM", "CAP_CONFIDENCE_ON_CAUSAL_LEAP", "REVISE_OR_REJECT_CLAIM"],
                    canonical_failure_code="F4",
                    canonical_failure_name="Causal Leap"
                )
                matched_patterns.append(fp_cau)
                reasons.append("Claim asserts direct causal mechanisms ('causes', 'leads to') while candidate evidence only reports observational correlation.")
                max_score = max(max_score, 0.88)

        # F5: Outdated Evidence fallback if not already in persistent patterns
        has_outdated = any(
            (meta.get("is_superseded") is True or meta.get("outdated") is True or (isinstance(meta.get("publication_year"), int) and 2026 - meta["publication_year"] > 20))
            for meta in [getattr(e, "metadata", {}) if hasattr(e, "metadata") else (e.get("metadata", {}) if isinstance(e, dict) else {}) for e in evidence]
        )
        if not any(p.canonical_failure_code == "F5" for p in matched_patterns):
            if has_outdated:
                fp_out = FailurePattern(
                    pattern_id="FP_OUT_011",
                    failure_type="OUTDATED_EVIDENCE",
                    failure_signature="FS-OUTDATED-EVIDENCE",
                    pattern_name="SUPERSEDED_HISTORICAL_EVIDENCE_USED",
                    severity="MEDIUM",
                    frequency=4,
                    trigger_conditions={"is_superseded": True},
                    observed_failure="Claim verified using obsolete evidence superseded by subsequent state-of-the-art findings.",
                    incorrect_behavior="Treating deprecated or superseded scientific findings as current proof.",
                    correct_behavior="Retrieve modern literature or qualify verdict as PARTIALLY_SUPPORTED with temporal limitation.",
                    prevention_rule="Depreciate evidence older than modern state of the art or marked superseded.",
                    adaptive_actions=["FLAG_OUTDATED_EVIDENCE", "RESTRICT_TEMPORAL_VALIDITY", "REVISE_OR_REJECT_CLAIM"],
                    canonical_failure_code="F5",
                    canonical_failure_name="Outdated Evidence"
                )
                matched_patterns.append(fp_out)
                reasons.append("Retrieved evidence is chronologically obsolete or marked as superseded by modern literature.")
                max_score = max(max_score, 0.82)

        matched = len(matched_patterns) > 0
        risk_level = "LOW"
        if max_score >= 0.85:
            risk_level = "VERY_HIGH" if max_score >= 0.92 else "HIGH"
        elif max_score >= 0.60:
            risk_level = "MEDIUM"

        # Hierarchy priority: F7 -> F6 -> F2 -> F4 -> F5 -> F1 -> F3 -> F8
        priority_order = {"F7": 1, "F6": 2, "F2": 3, "F4": 4, "F5": 5, "F1": 6, "F3": 7, "F8": 8}
        matched_patterns.sort(key=lambda p: priority_order.get(p.canonical_failure_code or "F1", 99))
        primary = matched_patterns[0] if matched_patterns else None
        adaptive_actions = []
        for pat in matched_patterns:
            for act in pat.adaptive_actions:
                if act not in adaptive_actions:
                    adaptive_actions.append(act)

        canonical_code = primary.canonical_failure_code if primary and primary.canonical_failure_code else "NONE"
        canonical_name = primary.canonical_failure_name if primary and primary.canonical_failure_name else ("Clean Verification Contract" if not matched else "Unspecified Failure")

        return PatternMatchResult(
            matched=matched,
            matched_pattern_ids=[p.pattern_id for p in matched_patterns],
            pattern_match_score=round(max_score, 4),
            risk_level=risk_level,
            primary_pattern=primary,
            canonical_failure_code=canonical_code,
            canonical_failure_name=canonical_name,
            adaptive_actions=adaptive_actions,
            prevention_triggered=False,
            claim_recheck_status="ACCEPT" if not matched else "REVISE",
            reasons=reasons,
            metadata={
                "evidence_count": len(evidence),
                "best_score": best_score,
                "contra_count": contra_count,
                "has_domain_mismatch": has_domain_mismatch
            }
        )

    # ----------------------------------------------------------------------
    # 3. ADAPTIVE VERIFICATION & CLAIM RE-CHECK
    # ----------------------------------------------------------------------

    def execute_claim_recheck(
        self,
        claim_text: str,
        evidence: List[Any],
        match_result: PatternMatchResult,
        current_verdict: str,
        current_confidence: float,
        question: Optional[str] = None
    ) -> Tuple[str, float, List[Any], PatternMatchResult]:
        """Performs Claim Re-Check under adaptive scrutiny.
        
        Outputs: (revised_verdict, revised_confidence, filtered_evidence, updated_match_result)
        Resolves action: ACCEPT / REVISE / REJECT.
        """
        if not match_result.matched or match_result.risk_level == "LOW":
            match_result.claim_recheck_status = "ACCEPT"
            match_result.prevention_triggered = False
            return current_verdict, current_confidence, evidence, match_result

        actions = match_result.adaptive_actions
        revised_verdict = current_verdict
        revised_conf = current_confidence
        filtered_evidence = list(evidence)
        recheck_action = "ACCEPT"

        # Action 1: Strict Domain Filtering & Exclusion of Off-Domain Passages (F6 / Population/Context Mismatch)
        if "REQUIRE_STRICT_DOMAIN_VALIDATION" in actions or "EXCLUDE_OFF_DOMAIN_PASSAGES" in actions or match_result.canonical_failure_code == "F6":
            clean_ev = []
            unrelated_terms = ["smart grid", "power grid", "electric grid", "drug discovery", "natural products", "carcinoma", "tumor"]
            q_lower = (question or "").lower()
            c_lower = claim_text.lower()
            is_quantum = "quantum" in q_lower or "quantum" in c_lower

            human_markers = ["in human", "human clinical", "cancer patients", "clinical patients", "humans", "patients", "human subjects"]
            preclin_markers = ["in vitro", "cell culture", "cell lines", "u87", "petri", "murine model", "mouse model", "rat model", "mice"]
            is_human_claim = any(hm in c_lower for hm in human_markers)

            for ev in filtered_evidence:
                t = str(getattr(ev, "text", "") or (ev.get("text") if isinstance(ev, dict) else "") or "").lower()
                meta = getattr(ev, "metadata", {}) if hasattr(ev, "metadata") else (ev.get("metadata", {}) if isinstance(ev, dict) else {})
                sc = float(getattr(ev, "score", 0.0) or (ev.get("score") if isinstance(ev, dict) else 0.0) or 0.0)

                is_off = False
                if is_quantum and any(u in t for u in unrelated_terms):
                    is_off = True
                if is_human_claim and any(pm in t for pm in preclin_markers) and not any(hm in t for hm in human_markers):
                    is_off = True
                if meta.get("evidence_eligible") is False or meta.get("topic_match") is False or sc <= 0.0:
                    is_off = True

                if not is_off:
                    clean_ev.append(ev)

            filtered_evidence = clean_ev
            if len(filtered_evidence) < len(evidence):
                match_result.reasons.append(f"Adaptive Re-Check excluded {len(evidence) - len(filtered_evidence)} off-domain or context-mismatched passages.")
                recheck_action = "REJECT"

        # Action 2: Enforce Relevance Floor 0.50 (F1 / F6)
        if "ENFORCE_RELEVANCE_FLOOR_050" in actions or "EXCLUDE_SUBTHRESHOLD_PASSAGES" in actions:
            clean_ev = []
            for ev in filtered_evidence:
                sc = float(getattr(ev, "score", 0.0) or (ev.get("score") if isinstance(ev, dict) else 0.0) or 0.0)
                if sc >= 0.50:
                    clean_ev.append(ev)
            if len(clean_ev) < len(filtered_evidence):
                match_result.reasons.append(f"Adaptive Re-Check excluded {len(filtered_evidence) - len(clean_ev)} sub-threshold passages (< 0.50).")
                filtered_evidence = clean_ev
                recheck_action = "REVISE"

        # Action 3: Sufficiency & Ungrounded Support Re-Check (F1 / Unsupported Claim)
        scores = [float(getattr(e, "score", 0.0) or (e.get("score") if isinstance(e, dict) else 0.0) or 0.0) for e in filtered_evidence]
        best_sc = max(scores, default=0.0)

        if len(filtered_evidence) == 0 or best_sc < 0.70:
            if str(revised_verdict).upper() in ("SUPPORT", "SUPPORTED"):
                revised_verdict = "NEI"
                revised_conf = 0.35
                recheck_action = "REJECT"
                match_result.reasons.append("Claim Re-Check REJECTED ungrounded SUPPORT verdict due to absence of qualifying >= 0.70 evidence.")
            elif len(filtered_evidence) == 0:
                revised_verdict = "NEI"
                revised_conf = 0.35
                recheck_action = "REJECT"
                match_result.reasons.append("Claim Re-Check REJECTED claim due to absence of eligible literature evidence.")

        # Action 4: Contradiction Re-Check & Debate Escalation (F7 / Contradictory Evidence)
        if "ESCALATE_TO_MULTI_AGENT_DEBATE" in actions or "CLAIM_RECHECK_CONTRADICTION" in actions or match_result.canonical_failure_code == "F7":
            has_contra = False
            for ev in filtered_evidence:
                rel = str(getattr(ev, "relation", "") or (ev.get("relation") if isinstance(ev, dict) else "") or "").upper()
                meta = getattr(ev, "metadata", {}) if hasattr(ev, "metadata") else (ev.get("metadata", {}) if isinstance(ev, dict) else {})
                rel_label = str(meta.get("relationship", "")).upper()
                if "CONTRADICT" in rel or "CONTRADICT" in rel_label or "REFUTE" in rel:
                    has_contra = True
                    break

            if has_contra and str(revised_verdict).upper() in ("SUPPORT", "SUPPORTED"):
                revised_verdict = "REFUTE"
                recheck_action = "REVISE"
                match_result.reasons.append("Claim Re-Check identified verified contradiction signals; revised verdict to REFUTE.")

        # Action 5: Dynamic Confidence Recalibration (F3 / Overgeneralization)
        if len(filtered_evidence) == 0:
            revised_conf = 0.35
        elif (revised_conf >= 0.85 and (len(filtered_evidence) < 2 or best_sc < 0.65)) or match_result.canonical_failure_code == "F3":
            revised_conf = round(min(0.60, max(0.35, best_sc * 0.75)), 4)
            match_result.reasons.append(f"Recalibrated confidence to {revised_conf*100:.0f}% to reflect limited evidence volume.")

        # Action 6: Causal Leap Qualification (F4 / Causal Leap)
        if "QUALIFY_CAUSAL_CLAIM" in actions or match_result.canonical_failure_code == "F4":
            if str(revised_verdict).upper() in ("SUPPORT", "SUPPORTED"):
                revised_verdict = "PARTIALLY_SUPPORTED"
                revised_conf = min(revised_conf, 0.60)
                recheck_action = "REVISE"
                match_result.reasons.append("Claim Re-Check qualified causal claim on observational evidence to PARTIALLY_SUPPORTED.")

        # Action 7: Outdated Evidence Depreciation (F5 / Outdated Evidence)
        if "FLAG_OUTDATED_EVIDENCE" in actions or match_result.canonical_failure_code == "F5":
            if str(revised_verdict).upper() in ("SUPPORT", "SUPPORTED"):
                revised_verdict = "PARTIALLY_SUPPORTED"
                revised_conf = min(revised_conf, 0.55)
                recheck_action = "REVISE"
                match_result.reasons.append("Claim Re-Check qualified verdict to PARTIALLY_SUPPORTED due to superseded or outdated chronological evidence.")

        # Action 8: Citation Mismatch & Misinterpretation (F2 / F8)
        if "VERIFY_EXACT_DOI_SOURCE" in actions or match_result.canonical_failure_code == "F2":
            revised_verdict = "INSUFFICIENT_EVIDENCE"
            revised_conf = 0.35
            recheck_action = "REJECT"
            match_result.reasons.append("Claim Re-Check REJECTED claim due to unverified bibliographic provenance or DOI citation mismatch.")
        elif match_result.canonical_failure_code == "F8":
            if str(revised_verdict).upper() in ("SUPPORT", "SUPPORTED"):
                revised_verdict = "PARTIALLY_SUPPORTED"
                revised_conf = min(revised_conf, 0.50)
                recheck_action = "REVISE"
                match_result.reasons.append("Claim Re-Check qualified verdict to PARTIALLY_SUPPORTED due to evidence misinterpretation or directional error.")

        match_result.claim_recheck_status = recheck_action
        match_result.prevention_triggered = True

        return revised_verdict, revised_conf, filtered_evidence, match_result
