"""Experiment Runner module for CiteGuard benchmark evaluation."""

import os
import json
import yaml
import time
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone

from src.evaluation.schema import (
    EvaluationExample,
    Prediction,
    EvaluationResult,
    MultiAgentMetrics
)
from src.evaluation.baselines import (
    BaseEvaluationSystem,
    MajorityBaseline,
    SparseBM25Baseline,
    DenseFAISSBaseline,
    HybridBaseline,
    MultiAgentBaseline
)
from src.evaluation.metrics import (
    compute_retrieval_metrics,
    compute_classification_metrics,
    compute_rhr_metrics,
    compute_calibration_ece
)
from src.evaluation.hallucination_ledger import HallucinationLedger
from src.evaluation.failure_analyzer import FailureAnalyzer
from src.evaluation.failure_signature import FailureSignature
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates scientific claim verification benchmark evaluations."""

    def __init__(self, config_filepath: str = "configs/evaluation.yaml"):
        self.config_filepath = config_filepath
        self.config = self.load_config(config_filepath)
        self.output_base_dir = self.config.get("experiment", {}).get("output_dir", "results/experiments")
        os.makedirs(self.output_base_dir, exist_ok=True)

    def load_config(self, filepath: str) -> Dict[str, Any]:
        """Load experiment configuration YAML."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_dataset_examples(
        self,
        dataset: str = "scifact",
        split: str = "test",
        limit: Optional[int] = None
    ) -> Tuple[List[EvaluationExample], bool]:
        """Load evaluation claim examples. Returns (examples, dataset_complete_flag)."""
        dataset = dataset.lower().strip()
        examples: List[EvaluationExample] = []
        is_complete = True

        # Path mapping
        candidate_paths = [
            f"data/processed/{dataset}/{dataset}_unified.jsonl",
            f"data/processed/{dataset}/claims_val.jsonl",
            f"data/processed/{dataset}/claims_train.jsonl"
        ]

        claims_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                claims_path = p
                break

        if not claims_path or not os.path.exists(claims_path):
            logger.warning(f"Claims file for dataset '{dataset}' not found at '{claims_path}'. Using synthetic dev fixture.")
            return self._get_synthetic_fixture_examples(dataset, split, limit), False

        raw_count = 0
        with open(claims_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_count += 1
                try:
                    data = json.loads(line)
                    ex_id = str(data.get("claim_id", data.get("id", f"ex_{len(examples)+1}")))
                    c_text = str(data.get("claim_text", data.get("claim", "")))
                    gt_lbl = str(data.get("ground_truth_label", data.get("label", "NEI")))
                    gt_docs = [str(d) for d in data.get("ground_truth_doc_ids", data.get("doc_ids", []))]

                    if c_text:
                        examples.append(
                            EvaluationExample(
                                example_id=ex_id,
                                claim_text=c_text,
                                dataset=dataset,
                                split=split,
                                ground_truth_label=gt_lbl,
                                ground_truth_doc_ids=gt_docs,
                                metadata=data.get("metadata", {})
                            )
                        )
                except Exception as e:
                    logger.warning(f"Error parsing line in {claims_path}: {e}")

        # Check completeness warning for small fixture datasets like CoVERT
        if raw_count < 10 and dataset in ("covert", "sciclaimhunt"):
            logger.warning(f"Dataset '{dataset}' contains only {raw_count} records. Marking as incomplete test fixture.")
            is_complete = False

        if limit and limit > 0:
            examples = examples[:limit]

        return examples, is_complete

    def _get_synthetic_fixture_examples(
        self,
        dataset: str,
        split: str,
        limit: Optional[int]
    ) -> List[EvaluationExample]:
        """Fallback small dev fixture for testing pipeline flow when datasets are missing."""
        fixtures = [
            EvaluationExample(
                example_id="dev_001",
                claim_text="MicroRNA-21 regulates apoptosis in human carcinoma cells.",
                dataset=dataset,
                split=split,
                ground_truth_label="SUPPORT",
                ground_truth_doc_ids=["doc_001"]
            ),
            EvaluationExample(
                example_id="dev_002",
                claim_text="CRISPR-Cas9 does not introduce off-target cleavage events.",
                dataset=dataset,
                split=split,
                ground_truth_label="REFUTE",
                ground_truth_doc_ids=["doc_002"]
            ),
            EvaluationExample(
                example_id="dev_003",
                claim_text="Vitamin C prevents all viral respiratory infections.",
                dataset=dataset,
                split=split,
                ground_truth_label="REFUTE",
                ground_truth_doc_ids=["doc_003"]
            ),
            EvaluationExample(
                example_id="dev_004",
                claim_text="BRCA1 mutations increase hereditary breast cancer risk.",
                dataset=dataset,
                split=split,
                ground_truth_label="SUPPORT",
                ground_truth_doc_ids=["doc_004"]
            ),
            EvaluationExample(
                example_id="dev_005",
                claim_text="Metformin extends lifespan in uninfected murine models.",
                dataset=dataset,
                split=split,
                ground_truth_label="NEI",
                ground_truth_doc_ids=[]
            ),
        ]
        return fixtures[:limit] if limit else fixtures

    def run_experiment(
        self,
        system_name: str = "citeguard_hybrid_multiagent",
        dataset: str = "scifact",
        split: str = "test",
        limit: Optional[int] = None,
        run_id_override: Optional[str] = None
    ) -> EvaluationResult:
        """Run complete benchmark evaluation experiment."""
        run_id = run_id_override or f"run_{system_name}_{dataset}_{int(time.time())}"
        run_dir = os.path.join(self.output_base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        logger.info(f"=== Starting Experiment Run: {run_id} ===")
        logger.info(f"System: {system_name} | Dataset: {dataset} | Limit: {limit}")

        examples, is_complete = self.load_dataset_examples(dataset=dataset, split=split, limit=limit)
        logger.info(f"Loaded {len(examples)} evaluation examples (Dataset Complete: {is_complete}).")

        # Instantiate system
        system = self._initialize_system(system_name)

        # Run predictions
        predictions: List[Prediction] = []
        start_eval_time = time.time()
        agent_calls_total = 0
        debate_activations = 0

        for idx, ex in enumerate(examples, 1):
            pred = system.predict(ex, run_id=run_id)
            predictions.append(pred)
            agent_calls_total += pred.agent_calls
            if pred.metadata.get("resolution_method") == "debate_resolution":
                debate_activations += 1

        total_exec_time = time.time() - start_eval_time

        # Compute Metrics
        gt_map = {ex.example_id: set(ex.ground_truth_doc_ids) for ex in examples if ex.ground_truth_doc_ids}
        retrieval_metrics = compute_retrieval_metrics(predictions, gt_map, k_list=[1, 3, 5, 10])
        classification_metrics = compute_classification_metrics(predictions)
        ece_score = compute_calibration_ece(predictions)

        # Compute Multi-Agent Metrics
        avg_exec_time = total_exec_time / float(len(examples)) if examples else 0.0
        debate_rate = debate_activations / float(len(examples)) if examples else 0.0
        multi_agent_metrics = MultiAgentMetrics(
            total_examples=len(examples),
            agents_invoked=agent_calls_total,
            reasoning_rounds=1.5 if debate_activations > 0 else 1.0,
            debate_activation_rate=debate_rate,
            average_execution_time_sec=avg_exec_time
        )

        # Compute Failure Signatures & Ledger Integration
        analyzer = FailureAnalyzer()
        current_signatures: List[FailureSignature] = []
        failure_breakdown: Dict[str, int] = {}

        for pred in predictions:
            gt_set = set(gt_map.get(pred.example_id, []))
            retrieved_res = [{"doc_id": d, "rank": r} for d, r in zip(pred.retrieved_doc_ids, pred.retrieved_ranks)]

            sig = analyzer.analyze_instance(
                claim_id=pred.example_id,
                claim_text=pred.claim_text,
                predicted_label=pred.predicted_label,
                ground_truth_label=pred.ground_truth_label,
                retrieved_results=retrieved_res,
                ground_truth_doc_ids=gt_set,
                dataset=dataset,
                split=split,
                run_id=run_id
            )

            if sig:
                current_signatures.append(sig)
                ft_val = sig.failure_type.value
                failure_breakdown[ft_val] = failure_breakdown.get(ft_val, 0) + 1

        # Load historical signatures from ledger for RHR calculation
        ledger_path = self.config.get("ledger", {}).get("ledger_path", "results/hallucination_ledger/hallucination_ledger.jsonl")
        ledger = HallucinationLedger(ledger_filepath=ledger_path)
        historical_sigs = ledger.signatures

        rhr_metrics = compute_rhr_metrics(current_signatures, historical_sigs)

        # Construct EvaluationResult
        result = EvaluationResult(
            run_id=run_id,
            system_name=system_name,
            dataset=dataset,
            split=split,
            total_examples=len(examples),
            retrieval_metrics=retrieval_metrics,
            classification_metrics=classification_metrics,
            multi_agent_metrics=multi_agent_metrics,
            rhr_metrics=rhr_metrics,
            failure_breakdown=failure_breakdown,
            metadata={
                "dataset_complete": is_complete,
                "limit": limit,
                "ece_calibration_error": ece_score,
                "development_run": limit is not None and limit < 100
            }
        )

        # Save artifacts to run directory
        self._save_run_artifacts(run_dir, result, predictions, current_signatures)

        logger.info(f"=== Experiment Completed: {run_id} ===")
        logger.info(f"Accuracy: {classification_metrics.accuracy:.4f} | Macro-F1: {classification_metrics.macro_f1:.4f} | MRR: {retrieval_metrics.mrr:.4f}")
        return result

    def _initialize_system(self, system_name: str) -> BaseEvaluationSystem:
        """Initialize requested evaluation baseline system."""
        system_name = system_name.lower().strip()

        if system_name == "majority_baseline":
            return MajorityBaseline()
        elif system_name == "bm25_sparse_baseline":
            bm25 = BM25Retriever()
            if os.path.exists("models/indexes/bm25_index.pkl"):
                bm25.load_index("models/indexes/bm25_index.pkl")
            return SparseBM25Baseline(bm25_retriever=bm25)
        elif system_name == "dense_faiss_baseline":
            dense = DenseRetriever()
            if os.path.exists("models/indexes/faiss_index.bin"):
                dense.load_index("models/indexes/faiss_index.bin", "models/indexes/doc_id_map.json")
            return DenseFAISSBaseline(dense_retriever=dense)
        elif system_name == "hybrid_rrf_baseline":
            hybrid = None
            if os.path.exists("models/indexes/bm25_index.pkl") and os.path.exists("models/indexes/faiss_index.bin"):
                bm25 = BM25Retriever()
                bm25.load_index("models/indexes/bm25_index.pkl")
                dense = DenseRetriever()
                dense.load_index("models/indexes/faiss_index.bin", "models/indexes/doc_id_map.json")
                hybrid = HybridRetriever(sparse_retriever=bm25, dense_retriever=dense)
            return HybridBaseline(hybrid_retriever=hybrid)
        elif system_name in ("citeguard_hybrid_multiagent", "multiagent"):
            return MultiAgentBaseline(provider_type=self.config.get("system", {}).get("provider_type", "mock"))
        else:
            logger.warning(f"Unknown system name '{system_name}'. Defaulting to MultiAgentBaseline.")
            return MultiAgentBaseline(provider_type="mock")

    def _save_run_artifacts(
        self,
        run_dir: str,
        result: EvaluationResult,
        predictions: List[Prediction],
        signatures: List[FailureSignature]
    ):
        """Save predictions.jsonl, metrics.json, errors.jsonl, and metadata.json under run directory."""
        # 1. Save config.yaml
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        # 2. Save predictions.jsonl
        with open(os.path.join(run_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p.to_dict()) + "\n")

        # 3. Save metrics.json
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            f.write(result.to_json())

        # 4. Save errors.jsonl
        with open(os.path.join(run_dir, "errors.jsonl"), "w", encoding="utf-8") as f:
            for sig in signatures:
                f.write(json.dumps(sig.to_dict()) + "\n")

        # 5. Save metadata.json
        metadata = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "system_name": result.system_name,
            "dataset": result.dataset,
            "split": result.split,
            "total_examples": result.total_examples,
            "development_run": result.metadata.get("development_run", False)
        }
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved run artifacts under: {run_dir}")
