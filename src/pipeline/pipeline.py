"""Top-level integrated pipeline module for CiteGuard scientific claim verification."""

import os
import time
import logging
from typing import List, Dict, Any, Optional

from src.utils.input_validation import validate_claim_input
from src.verification.schema import (
    VerificationRequest,
    RetrievedEvidence,
    VerificationResult,
    ExecutionMetrics,
    AgentDecision
)
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.policy import VerificationPolicy
from src.verification.providers.factory import get_provider
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class CiteGuardPipeline:
    """End-to-end integrated scientific claim verification pipeline."""

    def __init__(
        self,
        bm25_index_path: str = "models/indexes/bm25_index.pkl",
        faiss_index_path: str = "models/indexes/faiss_index.bin",
        map_path: str = "models/indexes/doc_id_map.json",
        provider_type: str = "mock",
        policy: Optional[VerificationPolicy] = None,
        ledger_path: str = "results/hallucination_ledger/hallucination_ledger.jsonl",
        runtime_ledger_path: str = "results/hallucination_ledger/runtime_audit.jsonl"
    ):
        self.bm25_index_path = bm25_index_path
        self.faiss_index_path = faiss_index_path
        self.map_path = map_path
        self.provider_type = provider_type
        self.policy = policy or VerificationPolicy()
        self.ledger_path = ledger_path
        self.runtime_ledger_path = runtime_ledger_path

        # Initialize Retriever if index files exist on disk
        self.retriever: Optional[HybridRetriever] = None
        self._init_retriever()

        # Initialize LLM Provider
        self.provider = get_provider(provider_type=self.provider_type, fallback_to_mock=True)

        # Initialize Multi-Agent Orchestrator
        self.orchestrator = MultiAgentOrchestrator(
            retriever=self.retriever,
            provider=self.provider,
            policy=self.policy,
            ledger_path=self.ledger_path,
            runtime_ledger_path=self.runtime_ledger_path
        )

    def _init_retriever(self):
        """Safely initialize hybrid retriever if vector and BM25 index files exist."""
        if (
            os.path.exists(self.bm25_index_path)
            and os.path.exists(self.faiss_index_path)
            and os.path.exists(self.map_path)
        ):
            try:
                bm25 = BM25Retriever()
                bm25.load_index(self.bm25_index_path)

                dense = DenseRetriever()
                dense.load_index(self.faiss_index_path, self.map_path)

                self.retriever = HybridRetriever(sparse_retriever=bm25, dense_retriever=dense)
                logger.info("CiteGuardPipeline initialized with active HybridRetriever.")
            except Exception as e:
                logger.warning(f"Could not load hybrid retrieval indices: {e}. Will use fallback evidence mode.")
                self.retriever = None
        else:
            logger.info("Retrieval index files not found. Pipeline operating in fallback mode.")
            self.retriever = None

    def verify(
        self,
        claim: Any,
        claim_id: str = "claim_001",
        dataset: str = "custom",
        split: str = "test",
        ground_truth_label: Optional[str] = None,
        ground_truth_doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        run_id: str = "run_default"
    ) -> VerificationResult:
        """Execute end-to-end validation, retrieval, verification, and failure analysis for a claim."""
        start_time = time.time()

        # Step 1: Input Validation & Normalization
        is_valid, normalized_claim, error_msg = validate_claim_input(claim)
        if not is_valid:
            logger.warning(f"Claim validation failed: {error_msg}")
            exec_time = time.time() - start_time
            metrics = ExecutionMetrics(total_execution_time_sec=exec_time, agent_calls_count=0)
            return VerificationResult(
                claim_id=claim_id,
                claim_text=str(claim) if claim is not None else "",
                final_label="ERROR",
                confidence=0.0,
                disagreement_status=False,
                resolution_method="input_validation_error",
                run_id=run_id,
                execution_metrics=metrics,
                metadata={"validation_error": error_msg}
            )

        # Step 2: Construct VerificationRequest
        gt_docs = ground_truth_doc_ids or []
        request = VerificationRequest(
            claim_id=claim_id,
            claim_text=normalized_claim,
            dataset=dataset,
            split=split,
            ground_truth_label=ground_truth_label,
            ground_truth_doc_ids=gt_docs,
            retrieval_mode=mode,
            top_k=top_k
        )

        # Fallback evidence if retriever is not active
        fallback_evidence = None
        if not self.orchestrator.retriever:
            fallback_evidence = [
                RetrievedEvidence(
                    doc_id="doc_fallback_001",
                    text=f"Retrieved passage relevant to {normalized_claim}",
                    score=0.85,
                    rank=1,
                    source="fallback_mock"
                )
            ]

        # Step 3: Run Orchestrator with Exception Guarding
        try:
            return self.orchestrator.verify(
                request=request,
                pre_retrieved_evidence=fallback_evidence,
                run_id=run_id
            )
        except Exception as e:
            logger.error(f"Pipeline exception during verification: {str(e)}", exc_info=True)
            exec_time = time.time() - start_time
            metrics = ExecutionMetrics(total_execution_time_sec=exec_time, agent_calls_count=0)
            return VerificationResult(
                claim_id=claim_id,
                claim_text=normalized_claim,
                final_label="ERROR",
                confidence=0.0,
                disagreement_status=False,
                resolution_method="pipeline_execution_error",
                run_id=run_id,
                execution_metrics=metrics,
                metadata={"pipeline_error": str(e)}
            )
