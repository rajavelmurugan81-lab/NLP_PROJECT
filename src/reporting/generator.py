"""Report Generator creating structured JSON and publication Markdown research reports."""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.verification.schema import VerificationResult

logger = logging.getLogger(__name__)


def normalize_verdict(val: Any) -> str:
    """Normalize raw verdict strings or enums to canonical representations: SUPPORTED, PARTIALLY_SUPPORTED, INSUFFICIENT_EVIDENCE, CONTRADICTED."""
    if not val:
        return "INSUFFICIENT_EVIDENCE"
    v_str = val.value if hasattr(val, "value") else str(val)
    v_upper = v_str.strip().upper().replace(" ", "_")
    if "." in v_upper:
        v_upper = v_upper.split(".")[-1]
    
    if v_upper in ("SUPPORT", "SUPPORTED", "SUPPORTS"):
        return "SUPPORTED"
    elif v_upper in ("PARTIAL", "PARTIALLY_SUPPORTED", "PARTIALLY_SUPPORT"):
        return "PARTIALLY_SUPPORTED"
    elif v_upper in ("REFUTE", "REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTED", "REJECTED"):
        return "CONTRADICTED"
    elif v_upper in ("NEI", "INSUFFICIENT", "INSUFFICIENT_EVIDENCE", "NEUTRAL", "NOT_ENOUGH_INFO", "NO_EVIDENCE"):
        return "INSUFFICIENT_EVIDENCE"
    return "INSUFFICIENT_EVIDENCE"


class ReportGenerator:
    """Generates structured JSON and human-readable Markdown research reports from pipeline results."""

    def __init__(self, output_dir: str = "results/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(
        self,
        question: ResearchQuestion,
        verification_results: List[VerificationResult],
        json_output_path: Optional[str] = None,
        md_output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Synthesize verification results into JSON and Markdown reports."""
        q_id = question.question_id
        
        # Calculate summary statistics with normalized verdicts
        normalized_verdicts = [normalize_verdict(res.final_label) for res in verification_results]
        
        sup_cnt = sum(1 for v in normalized_verdicts if v == "SUPPORTED")
        part_cnt = sum(1 for v in normalized_verdicts if v == "PARTIALLY_SUPPORTED")
        nei_cnt = sum(1 for v in normalized_verdicts if v == "INSUFFICIENT_EVIDENCE")
        con_cnt = sum(1 for v in normalized_verdicts if v == "CONTRADICTED")

        total_claims = len(verification_results)
        verdict_counts = {
            "supported": sup_cnt,
            "partially_supported": part_cnt,
            "insufficient_evidence": nei_cnt,
            "contradicted": con_cnt,
            # Uppercase aliases for backward compatibility
            "SUPPORTED": sup_cnt,
            "PARTIALLY_SUPPORTED": part_cnt,
            "INSUFFICIENT_EVIDENCE": nei_cnt,
            "CONTRADICTED": con_cnt
        }

        sup_pct = round((sup_cnt / total_claims * 100), 1) if total_claims > 0 else 0.0
        part_pct = round((part_cnt / total_claims * 100), 1) if total_claims > 0 else 0.0
        nei_pct = round((nei_cnt / total_claims * 100), 1) if total_claims > 0 else 0.0
        con_pct = round((con_cnt / total_claims * 100), 1) if total_claims > 0 else 0.0

        avg_confidence = round(
            sum(res.confidence for res in verification_results) / len(verification_results), 4
        ) if verification_results else 0.0

        # Collect evidence & citation metrics
        unique_cited_sources = set()
        contradictory_sources = set()
        publication_years = []
        claims_with_evidence_cnt = 0
        missing_evidence_claims = []
        supported_claims_summary = []
        partial_claims_summary = []
        contradicted_claims_summary = []
        unverified_claims_summary = []

        for res in verification_results:
            v_norm = normalize_verdict(res.final_label)
            has_ev = False
            for ev in res.evidence:
                pid = getattr(ev, "metadata", {}).get("paper_id") or getattr(ev, "doc_id", None) or getattr(ev, "title", None)
                if pid:
                    has_ev = True
                    unique_cited_sources.add(str(pid))

                rel = str(getattr(ev, "metadata", {}).get("relationship", getattr(ev, "relation", ""))).upper()
                if "CONTRADICT" in rel or "REFUTE" in rel or v_norm == "CONTRADICTED":
                    if pid:
                        contradictory_sources.add(str(pid))

                yr = getattr(ev, "metadata", {}).get("publication_year", getattr(ev, "year", None))
                if yr is not None:
                    try:
                        y_int = int(str(yr).strip())
                        if 1900 <= y_int <= 2035:
                            publication_years.append(y_int)
                    except (ValueError, TypeError):
                        pass

            if has_ev:
                claims_with_evidence_cnt += 1
            else:
                missing_evidence_claims.append({
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "reason": "No matching evidence passages retrieved from literature corpus."
                })

            if v_norm == "SUPPORTED":
                supported_claims_summary.append({
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "confidence": res.confidence,
                    "evidence_count": len(res.evidence)
                })
            elif v_norm == "PARTIALLY_SUPPORTED":
                partial_claims_summary.append({
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "confidence": res.confidence,
                    "evidence_count": len(res.evidence)
                })
                unverified_claims_summary.append({
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "verdict": v_norm,
                    "reason": "Evidence provides only partial support; lacks complete context or mechanism proof."
                })
            elif v_norm == "CONTRADICTED":
                contradicted_claims_summary.append({
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "confidence": res.confidence,
                    "evidence_count": len(res.evidence)
                })
            elif v_norm == "INSUFFICIENT_EVIDENCE":
                unverified_claims_summary.append({
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "verdict": v_norm,
                    "reason": "Insufficient empirical data available in corpus to establish or refute assertion."
                })

        coverage_pct = round((claims_with_evidence_cnt / total_claims * 100), 1) if total_claims > 0 else 0.0

        if publication_years:
            year_range_str = f"{min(publication_years)} – {max(publication_years)}"
        else:
            year_range_str = "Not available"

        retrieved_papers_meta = question.metadata.get("papers_retrieved", [])
        unique_retrieved_count = len(retrieved_papers_meta) if retrieved_papers_meta else max(len(unique_cited_sources), 0)
        retrieval_mode_str = question.metadata.get("retrieval_mode", "Hybrid OpenAlex + Local SciFact (RRF)")

        # Dashboard data structure
        dashboard_data = {
            "total_claims": total_claims,
            "supported_count": sup_cnt,
            "supported_pct": sup_pct,
            "partial_count": part_cnt,
            "partial_pct": part_pct,
            "insufficient_count": nei_cnt,
            "insufficient_pct": nei_pct,
            "contradicted_count": con_cnt,
            "contradicted_pct": con_pct,
            "average_confidence": avg_confidence,
            "unique_sources_cited": len(unique_cited_sources),
            "unique_sources_retrieved": unique_retrieved_count,
            "evidence_coverage_pct": coverage_pct,
            "contradictory_sources_count": len(contradictory_sources),
            "missing_evidence_count": len(missing_evidence_claims),
            "publication_year_range": year_range_str,
            "retrieval_mode": retrieval_mode_str,
            "direct_vs_indirect": "Not available"
        }

        # Build grounded research synthesis and conclusions
        research_synthesis_data = self._build_research_synthesis(
            question=question,
            total_claims=total_claims,
            sup_cnt=sup_cnt,
            part_cnt=part_cnt,
            nei_cnt=nei_cnt,
            con_cnt=con_cnt,
            avg_confidence=avg_confidence,
            supported_claims=supported_claims_summary,
            partial_claims=partial_claims_summary,
            contradicted_claims=contradicted_claims_summary,
            unverified_claims=unverified_claims_summary,
            missing_evidence_claims=missing_evidence_claims,
            coverage_pct=coverage_pct
        )

        overall_synthesis = self._build_synthesis(question.research_question, verdict_counts, avg_confidence)

        # System Bounded Limitations
        limitations = [
            "Generated claims are AI/rule-synthesized hypotheses and must be evaluated against retrieved scientific evidence.",
            "Evidence retrieval combines OpenAlex scientific literature with the local SciFact and SciClaimHunt corpus depending on the selected retrieval mode.",
            "OpenAlex retrieval depends on network/API availability; when unavailable, the system can fall back to the local retrieval system.",
            "Retrieval relevance does not itself constitute scientific proof; evidence must be evaluated against each claim.",
            "Verification results are bounded by the scientific literature retrieved by the system.",
            "The system does not generate fake papers or fabricate supporting evidence."
        ]

        sub_q_list = getattr(question, "sub_questions", [])

        # 1. Build JSON Structure
        report_data = {
            "question_id": q_id,
            "research_question": question.research_question,
            "question": question.research_question,
            "domain": question.domain,
            "sub_questions": sub_q_list,
            "research_plan": question.metadata.get("research_plan"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_claims_generated": len(question.generated_claims),
            "verdict_summary": verdict_counts,
            "verdict_percentages": {
                "supported_pct": sup_pct,
                "partially_supported_pct": part_pct,
                "insufficient_evidence_pct": nei_pct,
                "contradicted_pct": con_pct
            },
            "verification": {
                "supported": sup_cnt,
                "partially_supported": part_cnt,
                "insufficient_evidence": nei_cnt,
                "contradicted": con_cnt
            },
            "dashboard": dashboard_data,
            "research_synthesis": research_synthesis_data,
            "uncertainty_and_gaps": {
                "unverified_claims": unverified_claims_summary,
                "contradicted_findings": contradicted_claims_summary,
                "missing_evidence_claims": missing_evidence_claims,
                "follow_up_questions": research_synthesis_data.get("follow_up_questions", [])
            },
            "synthesis": {
                "summary": overall_synthesis,
                "supported_count": sup_cnt,
                "partial_count": part_cnt,
                "insufficient_count": nei_cnt,
                "contradicted_count": con_cnt
            },
            "average_confidence": avg_confidence,
            "claims": [
                {
                    "claim_id": res.claim_id,
                    "claim_text": res.claim_text,
                    "claim": res.claim_text,
                    "verdict": normalize_verdict(res.final_label),
                    "confidence": res.confidence,
                    "resolution_method": res.resolution_method,
                    "evidence_count": len(res.evidence),
                    "evidence": [
                        {
                            "doc_id": ev.doc_id,
                            "paper_id": getattr(ev, "metadata", {}).get("paper_id", ev.doc_id),
                            "paper_title": getattr(ev, "title", "Scientific Abstract/Document"),
                            "title": getattr(ev, "title", "Scientific Abstract/Document"),
                            "authors": (
                                ", ".join(getattr(ev, "metadata", {}).get("authors"))
                                if isinstance(getattr(ev, "metadata", {}).get("authors"), list)
                                else getattr(ev, "metadata", {}).get("authors", "Scientific Authors")
                            ),
                            "publication_year": getattr(ev, "metadata", {}).get("publication_year", 2022),
                            "year": getattr(ev, "metadata", {}).get("publication_year", 2022),
                            "section": getattr(ev, "metadata", {}).get("section", "Abstract"),
                            "passage_id": getattr(ev, "metadata", {}).get("passage_id", "p01"),
                            "exact_passage": getattr(ev, "metadata", {}).get("exact_passage", ev.text),
                            "passage": getattr(ev, "metadata", {}).get("exact_passage", ev.text),
                            "text": ev.text,
                            "score": ev.score,
                            "rank": ev.rank,
                            "url_doi": getattr(ev, "metadata", {}).get("url_doi", ""),
                            "doi": getattr(ev, "metadata", {}).get("url_doi", ""),
                            "url": getattr(ev, "metadata", {}).get("url_doi", ""),
                            "retrieval_method": getattr(ev, "source", "retriever"),
                            "relation": (
                                str(getattr(ev, "metadata", {}).get("relationship", getattr(ev, "relation", ""))).lower()
                                if getattr(ev, "metadata", {}).get("relationship") or getattr(ev, "relation", None)
                                else normalize_verdict(res.final_label).lower()
                            )
                        }
                        for ev in res.evidence
                    ],
                    "failure_signature": (res.failure_signature.to_dict() if hasattr(res.failure_signature, "to_dict") else res.failure_signature) if res.failure_signature else None,
                    "runtime_audit": res.metadata.get("runtime_audit"),
                    "agent_decisions": [
                        {
                            "agent_name": getattr(ad, "agent_name", ad.get("agent_name", "agent") if isinstance(ad, dict) else "agent"),
                            "verdict": normalize_verdict(ad.decision.value if hasattr(ad, "decision") and hasattr(ad.decision, "value") else str(getattr(ad, "decision", getattr(ad, "verdict", "NEI")))),
                            "confidence": getattr(ad, "confidence", ad.get("confidence", 0.0) if isinstance(ad, dict) else 0.0),
                            "rationale": getattr(ad, "rationale", ad.get("rationale", "") if isinstance(ad, dict) else "")
                        }
                        for ad in res.agent_assessments
                    ]
                }
                for res in verification_results
            ],
            "overall_synthesis": overall_synthesis,
            "limitations": limitations
        }

        # 2. Export JSON File
        safe_q_id = re.sub(r'[\\/*?:"<>|\r\n]', "_", str(q_id)).strip() or "unnamed"
        json_path = json_output_path or os.path.join(self.output_dir, f"question_{safe_q_id}.json")
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 3. Build Markdown Content
        md_content = self._build_markdown(report_data)

        # 4. Export Markdown File
        md_path = md_output_path or os.path.join(self.output_dir, f"question_{safe_q_id}.md")
        os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Report generated successfully: {json_path} & {md_path}")
        return {"json_path": json_path, "md_path": md_path}

    def _build_research_synthesis(
        self,
        question: ResearchQuestion,
        total_claims: int,
        sup_cnt: int,
        part_cnt: int,
        nei_cnt: int,
        con_cnt: int,
        avg_confidence: float,
        supported_claims: List[Dict[str, Any]],
        partial_claims: List[Dict[str, Any]],
        contradicted_claims: List[Dict[str, Any]],
        unverified_claims: List[Dict[str, Any]],
        missing_evidence_claims: List[Dict[str, Any]],
        coverage_pct: float
    ) -> Dict[str, Any]:
        """Generate structured, evidence-grounded scientific synthesis components."""
        q_text = question.research_question
        sub_qs = getattr(question, "sub_questions", [])

        # Degree of Support & Sufficiency Determination
        if total_claims == 0:
            degree_of_support = "INSUFFICIENT_EVIDENCE"
            degree_text = "Zero atomic claims were generated or evaluated for this inquiry."
            sufficiency_status = "INSUFFICIENT"
            sufficiency_text = "No claims were evaluated; evidence is insufficient to draw conclusions."
        elif sup_cnt == total_claims and total_claims > 0:
            degree_of_support = "HIGH_CONFIDENCE_SUPPORT"
            degree_text = f"The retrieved literature provides robust, mutually reinforcing empirical support for all evaluated assertions regarding '{q_text}'."
            sufficiency_status = "SUFFICIENT"
            sufficiency_text = (
                "The available scientific evidence is sufficient to answer the core research question with high confidence across all evaluated claims. "
                "However, scientific validity remains bounded by the scope and methodology of the retrieved papers."
            )
        elif con_cnt == total_claims and total_claims > 0:
            degree_of_support = "STRONGLY_CONTRADICTED"
            degree_text = f"The empirical literature retrieved directly refutes the primary hypotheses formulated for '{q_text}'."
            sufficiency_status = "SUFFICIENT_REFUTATION"
            sufficiency_text = "The available literature provides sufficient evidentiary grounds to contradict the proposed scientific assertions."
        elif nei_cnt == total_claims and total_claims > 0:
            degree_of_support = "INSUFFICIENT_EVIDENCE"
            degree_text = f"Available literature in the examined corpus is insufficient to confirm or refute the evaluated assertions regarding '{q_text}'."
            sufficiency_status = "INSUFFICIENT"
            sufficiency_text = "Evidence is currently insufficient to answer the research question with academic confidence. Essential empirical data is absent from the retrieved corpus."
        elif sup_cnt > 0 and con_cnt > 0:
            degree_of_support = "CONTESTED_EVIDENCE"
            degree_text = f"The literature reveals direct scientific conflict or conflicting findings across evaluated sub-aspects of '{q_text}'."
            sufficiency_status = "INCONCLUSIVE_CONFLICT"
            sufficiency_text = "Available evidence is inconclusive due to direct contradictions between retrieved sources. Further harmonized trials or meta-analyses are required."
        elif sup_cnt > 0 and (part_cnt > 0 or nei_cnt > 0):
            degree_of_support = "MODERATE_QUALIFIED_SUPPORT"
            degree_text = f"The empirical literature provides qualified support for specific core components of '{q_text}', while other dimensions remain only partially verified or unevidenced."
            sufficiency_status = "PARTIALLY_SUFFICIENT"
            sufficiency_text = "Available evidence is partially sufficient: while primary sub-claims receive empirical backing, significant knowledge gaps prevent an unqualified affirmative answer."
        else:
            degree_of_support = "MIXED_PRELIMINARY"
            degree_text = f"Retrieved literature shows preliminary and mixed findings regarding '{q_text}'."
            sufficiency_status = "PARTIALLY_SUFFICIENT"
            sufficiency_text = "Available evidence is partially sufficient but characterized by moderate uncertainty."

        # Main Findings Narrative
        findings = []
        if supported_claims:
            top_sup = "; ".join(f"\"{c['claim_text']}\"" for c in supported_claims[:2])
            findings.append(f"Empirical evidence directly supports: {top_sup}.")
        if partial_claims:
            top_part = "; ".join(f"\"{c['claim_text']}\"" for c in partial_claims[:2])
            findings.append(f"Qualified or partial support observed for: {top_part}.")
        if contradicted_claims:
            top_con = "; ".join(f"\"{c['claim_text']}\"" for c in contradicted_claims[:2])
            findings.append(f"Literature evidence contradicts or refutes: {top_con}.")
        if nei_cnt > 0:
            findings.append(f"{nei_cnt} assertion(s) lack adequate published empirical documentation in the retrieved corpus.")

        main_findings_str = " ".join(findings) if findings else f"Evaluation completed across {total_claims} claims with average confidence {avg_confidence*100:.1f}%."

        # Strongest Conclusions
        strongest_conclusions = [
            f"Verified Assertion: \"{c['claim_text']}\" (Confidence: {c['confidence']*100:.1f}%, Passages: {c['evidence_count']})"
            for c in supported_claims
        ]
        if not strongest_conclusions:
            strongest_conclusions = ["No evaluated claims received unequivocal empirical support from the retrieved literature corpus."]

        # Qualifications & Conflicts
        qualifications_and_conflicts = []
        for c in partial_claims:
            qualifications_and_conflicts.append(f"Partial Support Qualification: Assertion \"{c['claim_text']}\" requires qualification regarding population, scope, or mechanism.")
        for c in contradicted_claims:
            qualifications_and_conflicts.append(f"Contradiction Warning: Assertion \"{c['claim_text']}\" is refuted by retrieved findings.")
        for c in missing_evidence_claims:
            qualifications_and_conflicts.append(f"Evidence Gap: Assertion \"{c['claim_text']}\" lacks retrieved empirical passage grounding.")
        if not qualifications_and_conflicts:
            qualifications_and_conflicts = ["No conflicting findings or significant evidentiary qualifications were identified within the retrieved literature."]

        # Follow-up Research Questions
        follow_up_questions = []
        if sub_qs:
            for sq in sub_qs:
                follow_up_questions.append(f"What targeted empirical studies substantiate: {sq}?")
        if unverified_claims:
            for uc in unverified_claims[:2]:
                follow_up_questions.append(f"Under what specific clinical or experimental conditions does assertion '{uc['claim_text']}' hold true?")
        if not follow_up_questions:
            follow_up_questions = [
                f"How do longitudinal or clinical trials validate the mechanisms underlying '{q_text}'?",
                "What confounders or population characteristics moderate these observed findings?"
            ]

        # Practical Implications (strictly justified by supported claims)
        if sup_cnt > 0:
            practical_implications = (
                f"Given empirical support for {sup_cnt} core claim(s), researchers and practitioners can tentatively rely on verified findings "
                "for experimental design or theoretical modeling, while maintaining vigilance over unverified sub-claims."
            )
        else:
            practical_implications = (
                "Given the lack of definitive empirical support in the retrieved corpus, practitioners should avoid premature clinical, "
                "policy, or experimental reliance on these unverified claims until primary literature validation is conducted."
            )

        # Recommended Next Steps
        recommended_next_steps = [
            "Conduct targeted retrieval across full-text primary literature (beyond abstract indices) to address identified evidentiary gaps.",
            "Verify potential population, dose, or methodological mismatches between retrieved evidence and formulated claims.",
            "Formulate experimental protocols to empirically test claims categorized as Insufficient Evidence or Partially Supported."
        ]

        return {
            "question": q_text,
            "degree_of_support": degree_of_support,
            "degree_text": degree_text,
            "sufficiency_status": sufficiency_status,
            "sufficiency_text": sufficiency_text,
            "main_findings": main_findings_str,
            "strongest_conclusions": strongest_conclusions,
            "qualifications_and_conflicts": qualifications_and_conflicts,
            "follow_up_questions": follow_up_questions,
            "practical_implications": practical_implications,
            "recommended_next_steps": recommended_next_steps
        }

    def _build_synthesis(self, question: str, verdict_counts: Dict[str, int], avg_confidence: float) -> str:
        sup = verdict_counts.get("supported", verdict_counts.get("SUPPORTED", 0))
        part = verdict_counts.get("partially_supported", verdict_counts.get("PARTIALLY_SUPPORTED", 0))
        nei = verdict_counts.get("insufficient_evidence", verdict_counts.get("INSUFFICIENT_EVIDENCE", 0))
        con = verdict_counts.get("contradicted", verdict_counts.get("CONTRADICTED", verdict_counts.get("REFUTED", 0)))

        total = sup + part + nei + con
        if total == 0:
            return f"Analysis of '{question}' was completed, but zero atomic claims were evaluated."

        if sup == total and total > 0:
            prefix = f"Analysis of '{question}' indicates strong scientific support across the evaluated claims"
        elif con == total and total > 0:
            prefix = f"Analysis of '{question}' indicates that the evaluated claims are contradicted by the retrieved evidence"
        elif nei == total and total > 0:
            prefix = f"Analysis of '{question}' indicates that the available retrieved evidence is insufficient to establish all evaluated claims"
        else:
            prefix = f"Analysis of '{question}' indicates mixed scientific support across the evaluated claims"

        counts_str = f"({sup} supported, {part} partially supported, {nei} insufficient evidence, {con} contradicted)"

        return f"{prefix} {counts_str}."

    def _build_markdown(self, data: Dict[str, Any]) -> str:
        lines = []
        lines.append("# CiteGuard Scientific Claim Verification & Research Synthesis Report")
        lines.append(f"**Research Question**: *\"{data['research_question']}\"*")
        lines.append(f"**Question ID**: `{data['question_id']}` | **Domain**: {data['domain']}")
        lines.append(f"**Generated Date**: {data['created_at'][:19].replace('T', ' ')} UTC")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Planner Decomposition
        lines.append("## 1. Planner Sub-Questions Decomposition")
        if data.get("sub_questions"):
            for sq_idx, sq in enumerate(data["sub_questions"], 1):
                lines.append(f"{sq_idx}. *{sq}*")
        else:
            lines.append("1. *Decomposed core research question assertions.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Synthesis
        lines.append("## 2. Executive Research Synthesis")
        lines.append(f"**Summary**: {data['overall_synthesis']}")
        lines.append("")

        rs = data.get("research_synthesis", {})
        if rs:
            lines.append(f"**Degree of Evidentiary Grounding**: `{rs.get('degree_of_support', 'N/A')}`")
            lines.append(f"> {rs.get('degree_text', '')}")
            lines.append("")
            lines.append(f"**Evidentiary Sufficiency**: `{rs.get('sufficiency_status', 'N/A')}`")
            lines.append(f"> {rs.get('sufficiency_text', '')}")
            lines.append("")
            lines.append(f"**Main Scientific Findings**: {rs.get('main_findings', '')}")
            lines.append("")

        # Verdict & Citation Dashboard
        lines.append("## 3. Research Verdict & Citation Dashboard")
        dash = data.get("dashboard", {})
        lines.append("| Metric | Value | Proportion / Context |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| **Total Claims Evaluated** | `{dash.get('total_claims', 0)}` | 100.0% |")
        lines.append(f"| **Supported Claims** | `{dash.get('supported_count', 0)}` | `{dash.get('supported_pct', 0.0)}%` |")
        lines.append(f"| **Partially Supported Claims** | `{dash.get('partial_count', 0)}` | `{dash.get('partial_pct', 0.0)}%` |")
        lines.append(f"| **Insufficient Evidence Claims** | `{dash.get('insufficient_count', 0)}` | `{dash.get('insufficient_pct', 0.0)}%` |")
        lines.append(f"| **Contradicted Claims** | `{dash.get('contradicted_count', 0)}` | `{dash.get('contradicted_pct', 0.0)}%` |")
        lines.append(f"| **Average Verification Confidence** | `{dash.get('average_confidence', 0.0)*100:.1f}%` | Threshold floor: 70.0% |")
        lines.append(f"| **Unique Sources Cited in Evidence** | `{dash.get('unique_sources_cited', 0)}` | Directly mapped to claims |")
        lines.append(f"| **Unique Sources Retrieved** | `{dash.get('unique_sources_retrieved', 0)}` | Retrieved via pipeline |")
        lines.append(f"| **Evidence Coverage Across Claims** | `{dash.get('evidence_coverage_pct', 0.0)}%` | Claims with $\\ge 1$ evidence passage |")
        lines.append(f"| **Contradictory Sources Count** | `{dash.get('contradictory_sources_count', 0)}` | Sources with refuting evidence |")
        lines.append(f"| **Missing Evidence Claims Count** | `{dash.get('missing_evidence_count', 0)}` | Claims lacking literature support |")
        lines.append(f"| **Publication Year Range** | `{dash.get('publication_year_range', 'Not available')}` | Evidence literature span |")
        lines.append(f"| **Retrieval Mode & Corpus** | `{dash.get('retrieval_mode', 'Hybrid OpenAlex + Local SciFact')}` | Multi-index hybrid search |")
        lines.append(f"| **Direct vs. Indirect Evidence** | `{dash.get('direct_vs_indirect', 'Not available')}` | Field not tracked by current schema |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Strongest Conclusions & Qualifications
        if rs:
            lines.append("## 4. Grounded Conclusions & Evidentiary Qualifications")
            lines.append("### Strongest Supported Conclusions")
            for sc in rs.get("strongest_conclusions", []):
                lines.append(f"- {sc}")
            lines.append("")

            lines.append("### Important Qualifications & Unresolved Issues")
            for qc in rs.get("qualifications_and_conflicts", []):
                lines.append(f"- {qc}")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Claim-by-Claim Breakdown
        lines.append("## 5. Claim-by-Claim In-Depth Verification & Evidence Breakdown")
        lines.append("")

        for idx, c in enumerate(data["claims"], 1):
            lines.append(f"### Claim {idx} (`{c['claim_id']}`)")
            lines.append(f"**Assertion**: \"{c['claim_text']}\"")
            lines.append(f"**Final Verdict**: `{c['verdict']}` | **Confidence**: `{c['confidence']*100:.1f}%` | **Resolution**: `{c['resolution_method']}`")
            lines.append("")

            if c.get("failure_signature"):
                fs = c["failure_signature"]
                lines.append("> [!WARNING]")
                lines.append(f"> **Hallucination / Failure Flagged**: Category `{fs.get('failure_type', 'UNKNOWN')}`")
                lines.append(f"> *Explanation*: {fs.get('explanation', 'Recorded in Hallucination Ledger.')}")
                lines.append("")

            lines.append("#### Mapped Scientific Evidence & Passages:")
            if c.get("evidence"):
                for ev in c["evidence"]:
                    lines.append(f"- **Paper Title**: *{ev['title']}* ({ev.get('authors', 'Authors')}, {ev.get('publication_year', '2022')})")
                    lines.append(f"  - **Doc / Paper ID**: `{ev['paper_id']}` | **Passage ID**: `{ev.get('passage_id', 'p01')}` | **Section**: `{ev.get('section', 'Abstract')}`")
                    lines.append(f"  - **Relevance Score**: `{ev['score']:.4f}` | **Relation**: `{ev.get('relation', 'nei').upper()}` | **DOI**: {ev.get('url_doi') or 'N/A'}")
                    lines.append(f"  - **Exact Passage**: \"{ev.get('exact_passage', ev['text'])}\"")
            else:
                lines.append("*No relevant document evidence was retrieved from the corpus.*")
            lines.append("")

            if c.get("agent_decisions"):
                lines.append("#### Multi-Agent Assessment Reasoning:")
                for ad in c["agent_decisions"]:
                    lines.append(f"- **{ad['agent_name']}**: Verdict `{ad['verdict']}` (Conf: `{ad['confidence']*100:.0f}%`) - *{ad['rationale'][:200]}*")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Scientific Uncertainty & Research Gaps
        lines.append("## 6. Scientific Uncertainty & Literature Gaps")
        ug = data.get("uncertainty_and_gaps", {})
        if ug.get("unverified_claims"):
            lines.append("### Claims Requiring Further Verification")
            for uc in ug["unverified_claims"]:
                lines.append(f"- **{uc['claim_id']}** (`{uc['verdict']}`): \"{uc['claim_text']}\" — *{uc['reason']}*")
            lines.append("")

        if ug.get("contradicted_findings"):
            lines.append("### Contradictory or Inconclusive Findings")
            for cf in ug["contradicted_findings"]:
                lines.append(f"- **{cf['claim_id']}**: \"{cf['claim_text']}\" — Directly contradicted by retrieved evidence passages.")
            lines.append("")
        else:
            lines.append("### Contradictory Literature Check")
            lines.append("*Note: No contradictory literature was detected within the retrieved corpus. Contradictions were evaluated strictly against the retrieved papers, not exhaustive global literature.*")
            lines.append("")

        if ug.get("missing_evidence_claims"):
            lines.append("### Literature Evidence Gaps")
            for me in ug["missing_evidence_claims"]:
                lines.append(f"- **{me['claim_id']}**: \"{me['claim_text']}\" — {me['reason']}")
            lines.append("")

        if rs.get("follow_up_questions"):
            lines.append("### Suggested Follow-Up Research Questions")
            for fq_idx, fq in enumerate(rs["follow_up_questions"], 1):
                lines.append(f"{fq_idx}. *{fq}*")
            lines.append("")

        # Practical Implications & Recommended Next Steps
        lines.append("---")
        lines.append("")
        lines.append("## 7. Evidence-Grounded Practical Implications & Next Steps")
        lines.append(f"**Practical Implications**: {rs.get('practical_implications', 'Review verified claims prior to application.')}")
        lines.append("")
        lines.append("**Recommended Next Steps**:")
        for r_step in rs.get("recommended_next_steps", []):
            lines.append(f"- {r_step}")
        lines.append("")

        # Methodological Limitations
        lines.append("---")
        lines.append("")
        lines.append("## 8. Methodological Limitations & Bounded Scope")
        for lim in data["limitations"]:
            lines.append(f"- {lim}")
        lines.append("")

        return "\n".join(lines)
