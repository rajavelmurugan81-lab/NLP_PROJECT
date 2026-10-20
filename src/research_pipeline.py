"""Research Pipeline orchestrating Research Question to Verified Scientific Report workflow."""

import os
import sys
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

from src.claim_generation.generator import ClaimGenerator
from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.evaluation.claim_extractor import GroundedClaim, RetrievalFirstClaimExtractor
from src.retrieval.paper_source import UnifiedPaperRetriever
from src.retrieval.passage_extractor import PassageExtractor
from src.verification.evidence_mapper import EvidenceMapper
from src.verification.schema import VerificationRequest, VerificationResult, RetrievedEvidence
from src.pipeline import CiteGuardPipeline
from src.reporting.generator import ReportGenerator
from src.claim_generation.classifier import classify_question, QuestionType
from src.planner.planner_agent import PlannerAgent, PlannerResult
from src.retrieval.paper_evidence import check_paper_domain_relevance
from src.evaluation.runtime_auditor import RuntimeAuditor
from src.evaluation.failure_memory import FailureMemory

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Complete end-to-end Research Question -> Retrieval -> Grounded Claims -> Multi-Agent Verification Pipeline."""

    def __init__(
        self,
        provider_type: str = "mock",
        bm25_index_path: str = "models/indexes/bm25_index.pkl",
        faiss_index_path: str = "models/indexes/faiss_index.bin",
        map_path: str = "models/indexes/doc_id_map.json",
        output_dir: str = "results/reports",
        runtime_audit_path: Optional[str] = None,
        benchmark_ledger_path: Optional[str] = None
    ):
        self.provider_type = provider_type
        self.output_dir = output_dir
        self.runtime_audit_path = runtime_audit_path or os.path.join("results", "hallucination_ledger", "runtime_audit.jsonl")
        self.benchmark_ledger_path = benchmark_ledger_path or os.path.join("results", "hallucination_ledger", "hallucination_ledger.jsonl")

        # Initialize Planner Agent
        self.planner = PlannerAgent()

        # Initialize Claim Generator (retained as fallback when no literature is retrieved)
        self.claim_generator = ClaimGenerator(provider_type=self.provider_type)

        # Initialize Retrieval-First Claim Extractor
        self.claim_extractor = RetrievalFirstClaimExtractor()

        # Initialize Unified Paper Retriever
        self.paper_retriever = UnifiedPaperRetriever()

        # Initialize Passage Extractor
        self.passage_extractor = PassageExtractor()

        # Initialize Evidence Mapper
        self.evidence_mapper = EvidenceMapper()

        # Initialize Single-Claim CiteGuard Verification Pipeline (for backward compatibility & orchestrator access)
        self.verification_pipeline = CiteGuardPipeline(
            bm25_index_path=bm25_index_path,
            faiss_index_path=faiss_index_path,
            map_path=map_path,
            provider_type=self.provider_type,
            ledger_path=self.benchmark_ledger_path,
            runtime_ledger_path=self.runtime_audit_path
        )
        self.orchestrator = self.verification_pipeline.orchestrator

        # Initialize Report Generator
        self.report_generator = ReportGenerator(output_dir=self.output_dir)

        # Initialize Runtime Auditor
        self.runtime_auditor = RuntimeAuditor(ledger_filepath=self.runtime_audit_path)

        # Initialize Failure-Pattern Memory & Adaptive Verification Coordinator
        self.failure_memory = FailureMemory()

    def run_pipeline(
        self,
        research_question: str,
        question_id: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        run_id: str = "run_research",
        verbose: bool = True,
        max_claims: int = 3
    ) -> Dict[str, Any]:
        """Execute the end-to-end research question verification workflow."""
        start_time = time.time()
        effective_qid = question_id or f"q_{uuid.uuid4().hex[:8]}"

        # Step 1: Research Planning (PlannerAgent)
        if verbose:
            print("\n=== CiteGuard Research Pipeline ===")
            print(f"Research Question: \"{research_question}\"")
            print(f"LLM Provider: {self.provider_type} | Retrieval Mode: {mode} | Top-K: {top_k}\n")
            print("Step 1/6: Formulating Structured Research Plan (PlannerAgent)...")

        research_plan = self.planner.plan(research_question)
        q_class = classify_question(research_question)
        q_type_str = research_plan.question_type

        if verbose:
            print(f" -> Question Type: {q_type_str}")
            print(f" -> Research Objective: {research_plan.research_objective}")
            print(f" -> Formulated {len(research_plan.research_subquestions)} Subquestions & {len(research_plan.retrieval_queries)} Retrieval Queries")
            print("\nStep 2/6: Discovering Scientific Papers & Extracting Passage Evidence...")

        # Step 2: Paper Discovery & Passage Extraction (BEFORE CLAIM EXTRACTION)
        bibliographic_results: List[Dict[str, Any]] = []
        if q_class["question_type"] == QuestionType.BIBLIOGRAPHIC_SEARCH:
            search_year = q_class.get("target_year")
            sort_metric = "cited_by_count:desc" if q_class.get("ranking_metric") == "cited_by_count" else None
            papers = self.paper_retriever.search(
                query=q_class["topic"],
                top_k=max(top_k * 2, q_class.get("target_count") or 10),
                publication_year=search_year,
                sort_by=sort_metric
            )
            for b_idx, p in enumerate(papers, 1):
                p_doi = p.get("doi") or p.get("url_doi") or None
                if p_doi and isinstance(p_doi, str) and not p_doi.strip():
                    p_doi = None

                t_match = p.get("topic_match")
                if t_match is None:
                    t_match, _ = check_paper_domain_relevance(
                        query=research_question,
                        paper_text=p.get("abstract") or p.get("text") or "",
                        paper_title=p.get("title") or ""
                    )

                raw_cited = p.get("cited_by_count", 0)
                try:
                    c_count = int(raw_cited) if raw_cited is not None else 0
                except (ValueError, TypeError):
                    c_count = 0

                bibliographic_results.append({
                    "rank": b_idx,
                    "title": p.get("title", "Untitled Document"),
                    "authors": p.get("authors") or p.get("authors_list", "Scientific Authors"),
                    "publication_year": p.get("publication_year", 2026),
                    "citation_count": c_count,
                    "doi": p_doi,
                    "openalex_id": p.get("paper_id") or p.get("doc_id") or f"openalex_w_{b_idx}",
                    "relevance_score": float(p.get("relevance_score") or p.get("score") or 0.0),
                    "topic_match": bool(t_match),
                    "experimental_results_status": "NOT_VERIFIED"
                })
        else:
            papers = self.paper_retriever.search(query=research_question, top_k=top_k * 2)

        extracted_passages = self.passage_extractor.extract_passages_from_papers(papers)

        oa_status = getattr(self.paper_retriever.openalex_retriever, "last_status", "CONNECTED_WITH_RESULTS")
        oa_status_str = oa_status.value if hasattr(oa_status, "value") else str(oa_status)

        if verbose:
            print(f" -> Discovered {len(papers)} papers (OpenAlex Status: {oa_status_str}) and extracted {len(extracted_passages)} sentence passages.")
            print("\nStep 3/6: Retrieval-First Claim Extraction Grounded in Literature Passages...")

        # Step 3: Retrieval-First Claim Extraction Grounded in Retrieved Passages
        grounded_claims = self.claim_extractor.extract_claims(
            research_question=research_question,
            research_plan=research_plan,
            papers=papers,
            passages=extracted_passages,
            top_k=max_claims,
            question_id=effective_qid
        )

        domain_str = research_plan.domain_constraints[0] if research_plan.domain_constraints else "general_science"

        # Backward compatibility fallback: If retrieval found zero literature/passages, synthesize hypothesis claims
        if not grounded_claims:
            if verbose:
                print(" -> Note: No empirical passages retrieved; activating hypothesis claim fallback...")
            fallback_rq = self.claim_generator.generate_claims(research_question=research_question, question_id=effective_qid)
            final_claims = fallback_rq.generated_claims
        else:
            final_claims = grounded_claims

        rq = ResearchQuestion(
            question_id=effective_qid,
            research_question=research_question,
            domain=domain_str,
            sub_questions=list(research_plan.research_subquestions),
            generated_claims=final_claims,
            metadata={
                "research_plan": research_plan.to_dict(),
                "extraction_method": "retrieval_first_grounded" if grounded_claims else "hypothesis_fallback",
                "bibliographic_results": bibliographic_results
            }
        )

        if verbose:
            print(f" -> Prepared {len(rq.generated_claims)} Grounded Claims for verification:")
            for idx, c in enumerate(rq.generated_claims, 1):
                p_id_str = getattr(c, "passage_id", "N/A")
                print(f"    [{idx}] Claim ID: {c.claim_id} | Passage: {p_id_str}")
                print(f"        Text: \"{c.claim_text}\"")

        # Step 4: Claim-Evidence Mapping & Multi-Agent Verification
        if verbose:
            print("\nStep 4/6: Multi-Agent Adversarial Verification (SkepticAgent -> Verifier)...")

        verification_results: List[VerificationResult] = []
        runtime_audit_records = []

        for idx, claim in enumerate(rq.generated_claims, 1):
            if verbose:
                print(f" -> Verifying Claim {idx}/{len(rq.generated_claims)} (`{claim.claim_id}`)...")

            # Map claim to candidate passages
            links = self.evidence_mapper.map_claim_to_passages(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                passages=extracted_passages,
                top_k=top_k
            )

            is_sufficient, suff_metrics = self.evidence_mapper.evaluate_sufficiency(links)

            # Convert links to RetrievedEvidence for orchestrator
            pre_evidence: List[RetrievedEvidence] = []
            for l_idx, link in enumerate(links, 1):
                pre_evidence.append(
                    RetrievedEvidence(
                        doc_id=link.paper_id,
                        text=link.evidence_text,
                        score=link.evidence_score,
                        rank=l_idx,
                        source=link.source,
                        title=link.paper_title,
                        metadata={
                            "paper_id": link.paper_id,
                            "passage_id": link.passage_id,
                            "exact_passage": link.evidence_text,
                            "relationship": link.relationship,
                            "url_doi": link.doi,
                            "doi": link.doi,
                            "retrieval_score": link.retrieval_score,
                            "relevance_score": link.evidence_score,
                            "authors": getattr(link, "authors", "OpenAlex Scientific Authors"),
                            "publication_year": getattr(link, "publication_year", 2022)
                        }
                    )
                )

            # Ensure the grounded claim's exact passage is included at top if not already present
            if hasattr(claim, "exact_evidence_text") and claim.exact_evidence_text:
                has_exact = any(e.metadata.get("passage_id") == claim.passage_id for e in pre_evidence)
                if not has_exact:
                    pre_evidence.insert(0, RetrievedEvidence(
                        doc_id=claim.source_paper_id,
                        text=claim.exact_evidence_text,
                        score=claim.evidence_score,
                        rank=1,
                        source=claim.provenance.get("source", "openalex"),
                        title=claim.source_title,
                        metadata={
                            "paper_id": claim.source_paper_id,
                            "passage_id": claim.passage_id,
                            "exact_passage": claim.exact_evidence_text,
                            "relationship": "SUPPORTS",
                            "url_doi": claim.source_doi,
                            "doi": claim.source_doi,
                            "retrieval_score": claim.evidence_score,
                            "relevance_score": claim.evidence_score,
                            "authors": claim.metadata.get("authors", "Scientific Authors"),
                            "publication_year": claim.source_year or 2024
                        }
                    ))

            req = VerificationRequest(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                dataset="research_question",
                top_k=top_k,
                retrieval_mode=mode,
                metadata={
                    "managed_audit": True,
                    "question_type": q_type_str,
                    **(claim.metadata if hasattr(claim, "metadata") and isinstance(claim.metadata, dict) else {})
                }
            )

            res = self.orchestrator.verify(
                request=req,
                pre_retrieved_evidence=pre_evidence,
                run_id=f"{run_id}_{claim.claim_id}"
            )

            # Historical Failure Matching & Adaptive Risk Assessment
            match_res = self.failure_memory.match_historical_failures(
                claim_text=claim.claim_text,
                evidence=res.evidence,
                question=research_question,
                preliminary_verdict=res.final_label,
                confidence=res.confidence
            )

            # Adaptive Verification & Claim Re-Check
            if match_res.matched and match_res.risk_level in ("MEDIUM", "HIGH", "VERY_HIGH"):
                recheck_verdict, recheck_conf, recheck_ev, match_res = self.failure_memory.execute_claim_recheck(
                    claim_text=claim.claim_text,
                    evidence=res.evidence,
                    match_result=match_res,
                    current_verdict=res.final_label,
                    current_confidence=res.confidence,
                    question=research_question
                )
                res.final_label = recheck_verdict
                res.confidence = recheck_conf
                res.evidence = recheck_ev

            # Memory Gate: If verification passes cleanly without contract violation, record in Trusted Memory
            self.failure_memory.record_trusted_verification(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                verdict=res.final_label,
                confidence=res.confidence,
                evidence=res.evidence,
                question=research_question,
                pattern_matched=match_res.matched,
                recheck_status=match_res.claim_recheck_status,
                allow_partial=False
            )

            # Audit every claim through RuntimeAuditor and persist to runtime_audit.jsonl
            primary_act = match_res.adaptive_actions[0] if match_res.adaptive_actions else "NONE"
            audit_record = self.runtime_auditor.audit_claim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                verdict=res.final_label,
                confidence=res.confidence,
                evidence=res.evidence,
                question=research_question,
                question_type=q_type_str,
                retrieval_mode=mode,
                openalex_status=oa_status_str,
                local_retrieval_status="READY",
                run_id=run_id,
                matched_pattern_ids=match_res.matched_pattern_ids,
                pattern_match_score=match_res.pattern_match_score,
                risk_level=match_res.risk_level,
                adaptive_action=primary_act,
                prevention_triggered=match_res.prevention_triggered,
                claim_recheck_status=match_res.claim_recheck_status
            )
            runtime_audit_records.append(audit_record)

            # Attach audit, failure memory metadata, and grounded provenance to verification result
            res.metadata["runtime_audit"] = audit_record.to_dict()
            res.metadata["failure_pattern_match"] = match_res.to_dict()
            if hasattr(claim, "provenance"):
                res.metadata["claim_provenance"] = claim.provenance

            verification_results.append(res)
            
            # Update claim status
            claim.status = f"VERIFIED_{res.final_label}"

        # Step 5a: Closed-Loop Failure Learning — Automatic Pattern Re-Mining
        # If any claim in this query had a detected failure (non-CLEAN audit),
        # trigger pattern mining from the updated runtime audit ledger so that
        # newly recorded failures become reusable patterns for future queries.
        has_eligible_failure = any(
            r.to_dict().get("failure_type") not in (None, "NONE", "CLEAN", "")
            for r in runtime_audit_records
        )
        if has_eligible_failure:
            try:
                if verbose:
                    print("\nStep 5a/6: Closed-Loop Learning — Mining new failure patterns from runtime audit...")
                newly_mined = self.failure_memory.mine_patterns_from_historical_ledger()
                if verbose:
                    print(f" -> Re-mined {len(newly_mined)} failure patterns (updated failure_patterns.json)")
            except Exception as e:
                logger.warning(f"[ResearchPipeline] Pattern re-mining failed (non-fatal): {e}")

        # Step 5b: Hallucination Ledger Logging & Report Synthesis
        if verbose:
            print("\nStep 5/6: Auditing Claims & Logging to Runtime Hallucination Ledger...")
            print("Step 6/6: Generating Evidence-Grounded Research Synthesis Report...")

        report_files = self.report_generator.generate_report(
            question=rq,
            verification_results=verification_results
        )

        exec_time = time.time() - start_time

        if verbose:
            print("\nStep 6/6: Exporting Artifacts Completed!")
            print(f" -> Machine-readable JSON: {report_files['json_path']}")
            print(f" -> Human-readable Markdown: {report_files['md_path']}")
            print(f"Total Execution Time: {exec_time:.2f} seconds\n")

        return {
            "research_question": rq,
            "research_plan": research_plan.to_dict(),
            "verification_results": verification_results,
            "runtime_audit_records": [r.to_dict() for r in runtime_audit_records],
            "openalex_status": oa_status_str,
            "question_type": q_type_str,
            "papers_retrieved": papers,
            "passages_extracted": extracted_passages,
            "grounded_claims": [c.to_dict() if hasattr(c, "to_dict") else c for c in rq.generated_claims],
            "bibliographic_results": bibliographic_results,
            "pattern_mining_result": self.failure_memory.get_last_mining_result().to_dict() if self.failure_memory.get_last_mining_result() else None,
            "report_files": report_files,
            "total_execution_time_sec": exec_time
        }


_global_pipeline: Optional[ResearchPipeline] = None

def get_pipeline(provider: str = "mock") -> ResearchPipeline:
    """Module-level singleton getter for ResearchPipeline."""
    global _global_pipeline
    if _global_pipeline is None or _global_pipeline.provider_type != provider:
        _global_pipeline = ResearchPipeline(provider_type=provider)
    return _global_pipeline

