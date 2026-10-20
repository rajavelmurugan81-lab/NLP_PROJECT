"""Multi-Agent Verification Orchestrator for CiteGuard."""

import time
import logging
from typing import List, Dict, Any, Optional

from src.verification.schema import (
    VerificationRequest,
    RetrievedEvidence,
    AgentAssessment,
    AgentDecision,
    DebateResult,
    ExecutionMetrics,
    VerificationResult,
)
from src.verification.providers.base_provider import LLMProvider
from src.verification.providers.factory import get_provider
from src.verification.agents.evidence_agent import EvidenceAgent
from src.verification.agents.claim_analysis_agent import ClaimAnalysisAgent
from src.verification.agents.source_quality_agent import SourceQualityAgent
from src.verification.agents.verification_agent import VerificationAgent
from src.verification.skeptic_agent import SkepticAgent, SkepticResult
from src.verification.policy import VerificationPolicy
from src.verification.debate import DebateResolver
from src.retrieval.hybrid import HybridRetriever
from src.evaluation.failure_analyzer import FailureAnalyzer
from src.evaluation.hallucination_ledger import HallucinationLedger
from src.evaluation.failure_signature import FailureSignature
from src.evaluation.runtime_auditor import RuntimeAuditor, DEFAULT_RUNTIME_AUDIT_PATH

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates multi-agent scientific claim verification with adaptive routing and ledger integration."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        provider: Optional[LLMProvider] = None,
        policy: Optional[VerificationPolicy] = None,
        ledger_path: str = "results/hallucination_ledger/hallucination_ledger.jsonl",
        runtime_ledger_path: str = DEFAULT_RUNTIME_AUDIT_PATH,
        runtime_auditor: Optional[RuntimeAuditor] = None
    ):
        self.retriever = retriever
        self.provider = provider or get_provider("mock")
        self.policy = policy or VerificationPolicy()
        self.ledger_path = ledger_path
        self.runtime_ledger_path = runtime_ledger_path

        # Initialize agents
        self.evidence_agent = EvidenceAgent(provider=self.provider)
        self.claim_agent = ClaimAnalysisAgent(provider=self.provider)
        self.source_agent = SourceQualityAgent(provider=self.provider)
        self.skeptic_agent = SkepticAgent(provider=self.provider)
        self.verification_agent = VerificationAgent(provider=self.provider)
        self.debate_resolver = DebateResolver(provider=self.provider)

        # Failure analyzer and persistent ledgers (benchmark vs runtime audit)
        self.failure_analyzer = FailureAnalyzer()
        self.ledger = HallucinationLedger(ledger_filepath=self.ledger_path)
        self.runtime_auditor = runtime_auditor or RuntimeAuditor(ledger_filepath=self.runtime_ledger_path)

    def verify(
        self,
        request: VerificationRequest,
        pre_retrieved_evidence: Optional[List[RetrievedEvidence]] = None,
        run_id: str = "run_default"
    ) -> VerificationResult:
        """Execute end-to-end verification pipeline for a claim request."""
        start_time = time.time()
        agent_calls = 0
        retrieval_calls = 0

        # Ensure sub-agents use current orchestrator provider if dynamically changed
        if self.provider:
            for sub_agent in (self.evidence_agent, self.claim_agent, self.source_agent, self.skeptic_agent, self.verification_agent, self.debate_resolver):
                if hasattr(sub_agent, "provider") and sub_agent.provider != self.provider:
                    sub_agent.provider = self.provider

        # Step 1: Retrieve Evidence
        evidence: List[RetrievedEvidence] = []
        if pre_retrieved_evidence is not None:
            evidence = pre_retrieved_evidence
        elif self.retriever:
            retrieval_calls += 1
            try:
                retrieved_docs = self.retriever.retrieve(
                    query=request.claim_text,
                    top_k=request.top_k,
                    mode=request.retrieval_mode
                )
                for idx, doc in enumerate(retrieved_docs, 1):
                    meta = dict(doc.get("metadata", {}))
                    meta.setdefault("paper_id", doc.get("paper_id", doc.get("doc_id", f"paper_{idx}")))
                    meta.setdefault("authors", doc.get("authors", "Scientific Authors"))
                    meta.setdefault("publication_year", doc.get("publication_year", 2022))
                    meta.setdefault("section", doc.get("section", "Abstract"))
                    meta.setdefault("passage_id", doc.get("passage_id", f"p_{idx:02d}"))
                    meta.setdefault("exact_passage", doc.get("exact_passage", doc.get("text", "")))
                    meta.setdefault("url_doi", doc.get("url_doi", ""))

                    evidence.append(
                        RetrievedEvidence(
                            doc_id=str(doc.get("doc_id", f"doc_{idx}")),
                            text=str(doc.get("text", "")),
                            score=float(doc.get("score") or doc.get("rrf_score") or 0.0),
                            rank=idx,
                            title=str(doc.get("title", "Scientific Abstract/Document")),
                            source=str(doc.get("source", request.retrieval_mode)),
                            metadata=meta
                        )
                    )
            except Exception as e:
                logger.error(f"Retrieval error: {str(e)}")

        context = {
            "request": request,
            "claim_text": request.claim_text,
            "evidence": evidence
        }

        # Step 2: Check Evidence Sufficiency
        if not self.policy.is_evidence_sufficient(evidence):
            exec_time = time.time() - start_time
            metrics = ExecutionMetrics(
                total_execution_time_sec=exec_time,
                agent_calls_count=0,
                reasoning_rounds=1,
                retrieval_calls_count=retrieval_calls,
                token_usage=dict(self.provider.token_usage)
            )
            # Dynamic confidence for insufficient evidence:
            # If zero evidence was retrieved, confidence in claim support is low (0.30 - 0.40)
            # reflecting absence of evidence rather than a confident 90%
            if not evidence or len(evidence) == 0:
                dynamic_conf = 0.35
            else:
                max_sc = max((float(getattr(e, "score", 0.0) or 0.0) for e in evidence), default=0.0)
                dynamic_conf = round(min(0.55, max(0.25, max_sc * 0.5)), 4)

            return self._build_result(
                request=request,
                final_label="NEI",
                confidence=dynamic_conf,
                evidence=evidence,
                assessments=[],
                disagreement_status=False,
                resolution_method="insufficient_evidence_fallback",
                debate_result=None,
                run_id=run_id,
                metrics=metrics
            )

        # Step 3: Run Agents
        assessments: List[AgentAssessment] = []
        # Step 3a: Initial assessments
        for agent in [self.evidence_agent, self.claim_agent, self.source_agent]:
            a_res = agent.analyze(context)
            assessments.append(a_res)
            agent_calls += 1

        # Step 3b: SkepticAgent runs adversarially with prior assessments in context
        context["prior_assessments"] = list(assessments)
        skeptic_res = self.skeptic_agent.analyze(context)
        assessments.append(skeptic_res)
        agent_calls += 1

        # Step 3c: Inject skeptic result into context for VerificationAgent
        context["skeptic_result"] = skeptic_res.metadata.get("skeptic_result")
        v_res = self.verification_agent.analyze(context)
        assessments.append(v_res)
        agent_calls += 1

        # Step 4: Check Agreement & Adaptive Routing
        agreement_info = self.policy.evaluate_agreement(assessments)
        should_debate = self.policy.should_trigger_debate(assessments)

        debate_result: Optional[DebateResult] = None
        resolution_method = "direct_consensus"

        if should_debate:
            resolution_method = "debate_resolution"
            debate_result = self.debate_resolver.resolve(
                claim_text=request.claim_text,
                evidence=evidence,
                assessments=assessments,
                policy=self.policy
            )
            agent_calls += 1
            final_decision = debate_result.final_decision
            base_agent_conf = (
                sum(a.confidence for a in debate_result.revised_assessments) / float(len(debate_result.revised_assessments))
                if debate_result.revised_assessments else 0.5
            )
        else:
            final_decision = agreement_info["majority_decision"]
            base_agent_conf = sum(a.confidence for a in assessments) / float(len(assessments)) if assessments else 0.5

        # Step 4.1: Compute Dynamic Grounded Confidence
        # Derived from best evidence relevance, average relevance, passage count, domain validity, and agent agreement
        ev_scores = [float(getattr(e, "score", 0.0) or 0.0) for e in evidence if getattr(e, "score", None) is not None]
        best_ev = max(ev_scores, default=0.0)
        avg_ev = sum(ev_scores) / len(ev_scores) if ev_scores else 0.0
        num_passages = len(ev_scores)

        supp_count = sum(1 for a in assessments if a.decision == AgentDecision.SUPPORT)
        contra_count = sum(1 for a in assessments if a.decision == AgentDecision.REFUTE)
        nei_count = sum(1 for a in assessments if a.decision == AgentDecision.NEI)
        total_ass = len(assessments) or 1
        agreement_factor = max(supp_count, contra_count, nei_count) / float(total_ass)

        # Evidence validity check: if no evidence, out-of-domain evidence (best_ev <= 0.40), or NEI verdict
        if not evidence or best_ev <= 0.40 or final_decision == AgentDecision.NEI:
            confidence = round(min(0.35, max(0.20, best_ev * 0.5)), 4)
        else:
            ev_factor = (0.50 * best_ev) + (0.30 * avg_ev) + (0.20 * min(1.0, num_passages / 3.0))
            raw_conf = (base_agent_conf * 0.35) + (ev_factor * 0.50) + (agreement_factor * 0.15)
            if contra_count > 0 and final_decision == AgentDecision.SUPPORT:
                raw_conf *= 0.80
            confidence = round(min(0.96, max(0.40, raw_conf)), 4)

        final_label = final_decision.value

        exec_time = time.time() - start_time
        metrics = ExecutionMetrics(
            total_execution_time_sec=exec_time,
            agent_calls_count=agent_calls,
            reasoning_rounds=2 if should_debate else 1,
            retrieval_calls_count=retrieval_calls,
            token_usage=dict(self.provider.token_usage)
        )

        return self._build_result(
            request=request,
            final_label=final_label,
            confidence=round(confidence, 4),
            evidence=evidence,
            assessments=assessments,
            disagreement_status=agreement_info["disagreement_status"],
            resolution_method=resolution_method,
            debate_result=debate_result,
            run_id=run_id,
            metrics=metrics
        )

    def _build_result(
        self,
        request: VerificationRequest,
        final_label: str,
        confidence: float,
        evidence: List[RetrievedEvidence],
        assessments: List[AgentAssessment],
        disagreement_status: bool,
        resolution_method: str,
        debate_result: Optional[DebateResult],
        run_id: str,
        metrics: ExecutionMetrics
    ) -> VerificationResult:
        failure_sig_dict: Optional[Dict[str, Any]] = None

        # Step 5: Phase 5 Hallucination Ledger Integration
        if request.ground_truth_label:
            gt_label_norm = str(request.ground_truth_label).strip().upper()
            pred_label_norm = str(final_label).strip().upper()

            # Normalize GT label synonyms
            if gt_label_norm in ("SUPPORTED", "SUPPORTS"):
                gt_label_norm = "SUPPORT"
            elif gt_label_norm in ("REFUTED", "REFUTES", "CONTRADICT"):
                gt_label_norm = "REFUTE"
            elif gt_label_norm in ("NEUTRAL", "INSUFFICIENT"):
                gt_label_norm = "NEI"

            if pred_label_norm != gt_label_norm:
                retrieved_results = [
                    {"doc_id": e.doc_id, "rank": e.rank}
                    for e in evidence
                ]
                gt_doc_set = set(request.ground_truth_doc_ids) if request.ground_truth_doc_ids else set()

                sig = self.failure_analyzer.analyze_instance(
                    claim_id=request.claim_id,
                    claim_text=request.claim_text,
                    predicted_label=pred_label_norm,
                    ground_truth_label=gt_label_norm,
                    retrieved_results=retrieved_results,
                    ground_truth_doc_ids=gt_doc_set,
                    dataset=request.dataset,
                    split=request.split,
                    retrieval_mode=request.retrieval_mode,
                    run_id=run_id
                )

                if sig:
                    self.ledger.add_signature(sig)
                    failure_sig_dict = sig.to_dict()
        else:
            # Step 5b: Runtime Failure Signature Analysis for Live Research Questions
            sig = self.failure_analyzer.analyze_runtime_instance(
                claim_id=request.claim_id,
                claim_text=request.claim_text,
                predicted_label=final_label,
                confidence=confidence,
                evidence=evidence,
                disagreement_status=disagreement_status,
                resolution_method=resolution_method,
                dataset=request.dataset,
                retrieval_mode=request.retrieval_mode,
                run_id=run_id,
                question=request.claim_text
            )
            if sig:
                failure_sig_dict = sig.to_dict()
                # Route live runtime failure signatures strictly to runtime audit ledger, NEVER to benchmark ledger
                # Avoid duplicate logging if caller (e.g. ResearchPipeline) manages the full runtime audit
                is_managed = bool(request.metadata and request.metadata.get("managed_audit"))
                if not is_managed:
                    self.runtime_auditor.audit_claim(
                        claim_id=request.claim_id,
                        claim_text=request.claim_text,
                        verdict=final_label,
                        confidence=confidence,
                        evidence=evidence,
                        question=request.claim_text,
                        question_type=str(request.metadata.get("question_type", "SCIENTIFIC_HYPOTHESIS")) if request.metadata else "SCIENTIFIC_HYPOTHESIS",
                        retrieval_mode=request.retrieval_mode,
                        run_id=run_id
                    )

        res_metadata = dict(request.metadata or {})
        sk_data = next((a.metadata.get("skeptic_result") for a in assessments if a.agent_name == "SkepticAgent" and a.metadata.get("skeptic_result")), None)
        if sk_data:
            res_metadata["skeptic_result"] = sk_data

        return VerificationResult(
            claim_id=request.claim_id,
            claim_text=request.claim_text,
            final_label=final_label,
            confidence=confidence,
            evidence=evidence,
            agent_assessments=assessments,
            disagreement_status=disagreement_status,
            resolution_method=resolution_method,
            debate_result=debate_result,
            failure_signature=failure_sig_dict,
            run_id=run_id,
            execution_metrics=metrics,
            metadata=res_metadata
        )
