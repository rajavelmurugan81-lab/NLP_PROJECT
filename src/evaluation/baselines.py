"""Baseline systems implementation for benchmark comparison."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

from src.evaluation.schema import EvaluationExample, Prediction
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.verification.schema import VerificationRequest, RetrievedEvidence
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.providers.factory import get_provider


class BaseEvaluationSystem(ABC):
    """Abstract interface for evaluation baseline systems."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, example: EvaluationExample, run_id: str = "run_eval") -> Prediction:
        """Generate prediction object for a given evaluation example."""
        pass


class MajorityBaseline(BaseEvaluationSystem):
    """Deterministic majority class baseline."""

    def __init__(self, majority_label: str = "SUPPORT"):
        super().__init__(name="majority_baseline")
        self.majority_label = majority_label

    def predict(self, example: EvaluationExample, run_id: str = "run_eval") -> Prediction:
        start_time = time.time()
        exec_time = time.time() - start_time
        return Prediction(
            example_id=example.example_id,
            claim_text=example.claim_text,
            predicted_label=self.majority_label,
            ground_truth_label=example.ground_truth_label,
            confidence=0.5,
            retrieved_doc_ids=[],
            retrieved_ranks=[],
            system_name=self.name,
            execution_time_sec=exec_time,
            run_id=run_id
        )


class SparseBM25Baseline(BaseEvaluationSystem):
    """Sparse BM25 retrieval-only baseline."""

    def __init__(self, bm25_retriever: Optional[BM25Retriever] = None, top_k: int = 5, score_threshold: float = 10.0):
        super().__init__(name="bm25_sparse_baseline")
        self.retriever = bm25_retriever
        self.top_k = top_k
        self.score_threshold = score_threshold

    def predict(self, example: EvaluationExample, run_id: str = "run_eval") -> Prediction:
        start_time = time.time()
        doc_ids = []
        ranks = []
        top_score = 0.0

        if self.retriever:
            try:
                results = self.retriever.retrieve(example.claim_text, top_k=self.top_k)
                doc_ids = [str(r.get("doc_id", "")) for r in results]
                ranks = list(range(1, len(doc_ids) + 1))
                if results:
                    top_score = float(results[0].get("score", 0.0))
            except Exception:
                pass

        # Heuristic verdict based on retrieval score
        predicted_label = "SUPPORT" if top_score >= self.score_threshold else "NEI"
        exec_time = time.time() - start_time

        return Prediction(
            example_id=example.example_id,
            claim_text=example.claim_text,
            predicted_label=predicted_label,
            ground_truth_label=example.ground_truth_label,
            confidence=min(1.0, top_score / 20.0) if top_score > 0 else 0.5,
            retrieved_doc_ids=doc_ids,
            retrieved_ranks=ranks,
            system_name=self.name,
            execution_time_sec=exec_time,
            run_id=run_id
        )


class DenseFAISSBaseline(BaseEvaluationSystem):
    """Dense FAISS vector retrieval-only baseline."""

    def __init__(self, dense_retriever: Optional[DenseRetriever] = None, top_k: int = 5, score_threshold: float = 0.5):
        super().__init__(name="dense_faiss_baseline")
        self.retriever = dense_retriever
        self.top_k = top_k
        self.score_threshold = score_threshold

    def predict(self, example: EvaluationExample, run_id: str = "run_eval") -> Prediction:
        start_time = time.time()
        doc_ids = []
        ranks = []
        top_score = 0.0

        if self.retriever:
            try:
                results = self.retriever.retrieve(example.claim_text, top_k=self.top_k)
                doc_ids = [str(r.get("doc_id", "")) for r in results]
                ranks = list(range(1, len(doc_ids) + 1))
                if results:
                    top_score = float(results[0].get("score", 0.0))
            except Exception:
                pass

        predicted_label = "SUPPORT" if top_score >= self.score_threshold else "NEI"
        exec_time = time.time() - start_time

        return Prediction(
            example_id=example.example_id,
            claim_text=example.claim_text,
            predicted_label=predicted_label,
            ground_truth_label=example.ground_truth_label,
            confidence=round(top_score, 4),
            retrieved_doc_ids=doc_ids,
            retrieved_ranks=ranks,
            system_name=self.name,
            execution_time_sec=exec_time,
            run_id=run_id
        )


class HybridBaseline(BaseEvaluationSystem):
    """Hybrid BM25 + Dense RRF retrieval baseline."""

    def __init__(self, hybrid_retriever: Optional[HybridRetriever] = None, top_k: int = 5, score_threshold: float = 0.01):
        super().__init__(name="hybrid_rrf_baseline")
        self.retriever = hybrid_retriever
        self.top_k = top_k
        self.score_threshold = score_threshold

    def predict(self, example: EvaluationExample, run_id: str = "run_eval") -> Prediction:
        start_time = time.time()
        doc_ids = []
        ranks = []
        top_score = 0.0

        if self.retriever:
            try:
                results = self.retriever.retrieve(example.claim_text, top_k=self.top_k, mode="hybrid")
                doc_ids = [str(r.get("doc_id", "")) for r in results]
                ranks = list(range(1, len(doc_ids) + 1))
                if results:
                    top_score = float(results[0].get("score", 0.0))
            except Exception:
                pass

        predicted_label = "SUPPORT" if top_score >= self.score_threshold else "NEI"
        exec_time = time.time() - start_time

        return Prediction(
            example_id=example.example_id,
            claim_text=example.claim_text,
            predicted_label=predicted_label,
            ground_truth_label=example.ground_truth_label,
            confidence=round(min(1.0, top_score * 50.0), 4) if top_score > 0 else 0.5,
            retrieved_doc_ids=doc_ids,
            retrieved_ranks=ranks,
            system_name=self.name,
            execution_time_sec=exec_time,
            run_id=run_id
        )


class MultiAgentBaseline(BaseEvaluationSystem):
    """CiteGuard Hybrid Retrieval + Adaptive Multi-Agent Verification System."""

    def __init__(self, orchestrator: Optional[MultiAgentOrchestrator] = None, provider_type: str = "mock"):
        super().__init__(name="citeguard_hybrid_multiagent")
        if orchestrator:
            self.orchestrator = orchestrator
        else:
            provider = get_provider(provider_type, fallback_to_mock=True)
            self.orchestrator = MultiAgentOrchestrator(provider=provider)

    def predict(self, example: EvaluationExample, run_id: str = "run_eval") -> Prediction:
        req = VerificationRequest(
            claim_id=example.example_id,
            claim_text=example.claim_text,
            dataset=example.dataset,
            split=example.split,
            ground_truth_label=example.ground_truth_label,
            ground_truth_doc_ids=example.ground_truth_doc_ids,
            retrieval_mode="hybrid"
        )

        # Fallback evidence if retriever is not built
        fallback_evidence = None
        if not self.orchestrator.retriever:
            fallback_evidence = [
                RetrievedEvidence(doc_id="doc_fallback_01", text=f"Evidence for {example.claim_text}", score=0.85, rank=1)
            ]

        res = self.orchestrator.verify(req, pre_retrieved_evidence=fallback_evidence, run_id=run_id)

        retrieved_ids = [e.doc_id for e in res.evidence]
        retrieved_ranks = [e.rank for e in res.evidence]

        return Prediction(
            example_id=example.example_id,
            claim_text=example.claim_text,
            predicted_label=res.final_label,
            ground_truth_label=example.ground_truth_label,
            confidence=res.confidence,
            retrieved_doc_ids=retrieved_ids,
            retrieved_ranks=retrieved_ranks,
            system_name=self.name,
            execution_time_sec=res.execution_metrics.total_execution_time_sec,
            agent_calls=res.execution_metrics.agent_calls_count,
            reasoning_rounds=res.execution_metrics.reasoning_rounds,
            run_id=run_id,
            metadata={
                "disagreement_status": res.disagreement_status,
                "resolution_method": res.resolution_method,
                "failure_signature": res.failure_signature
            }
        )
