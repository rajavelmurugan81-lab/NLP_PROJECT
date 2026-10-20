"""CLI demonstration script for Phase 6 Multi-Agent Scientific Claim Verification."""

import argparse
import json
import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.verification.schema import VerificationRequest, RetrievedEvidence
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.providers.factory import get_provider
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_claim")


def main():
    parser = argparse.ArgumentParser(description="CiteGuard Multi-Agent Claim Verification CLI")
    parser.add_argument("--claim", type=str, required=True, help="Scientific claim text to verify")
    parser.add_argument("--claim-id", type=str, default="cli_claim_001", help="Claim Identifier")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset tag")
    parser.add_argument("--retrieval-mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"], help="Retrieval mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence items to retrieve")
    parser.add_argument("--provider", type=str, default="mock", help="LLM Provider type ('mock' or 'openai')")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional ground truth label for failure logging")
    parser.add_argument("--gt-doc-ids", type=str, default=None, help="Comma-separated list of ground truth doc IDs")
    args = parser.parse_args()

    logger.info("=== Phase 6: Multi-Agent Claim Verification Pipeline ===")
    logger.info(f"Target Claim: \"{args.claim}\"")
    logger.info(f"Retrieval Mode: {args.retrieval_mode} | Provider: {args.provider}")

    # Load retriever if index files exist
    retriever = None
    bm25_path = "models/indexes/bm25_index.pkl"
    faiss_path = "models/indexes/faiss_index.bin"
    map_path = "models/indexes/doc_id_map.json"

    if os.path.exists(bm25_path) and os.path.exists(faiss_path) and os.path.exists(map_path):
        try:
            logger.info("Loading hybrid retrieval indices...")
            bm25 = BM25Retriever()
            bm25.load_index(bm25_path)

            dense = DenseRetriever()
            dense.load_index(faiss_path, map_path)

            retriever = HybridRetriever(sparse_retriever=bm25, dense_retriever=dense)
            logger.info("Hybrid retriever loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load retrieval indices: {e}. Will use fallback evidence.")
    else:
        logger.info("Retrieval indices not found. Using fallback mock evidence for verification test.")

    # Fallback evidence if retriever is unavailable
    fallback_evidence = None
    if not retriever:
        fallback_evidence = [
            RetrievedEvidence(
                doc_id="doc_fallback_001",
                text="Scientific study confirms microRNA-21 regulates cellular apoptosis in carcinoma models.",
                score=0.89,
                rank=1,
                source="fallback_mock"
            )
        ]

    # Instantiate provider
    provider = get_provider(provider_type=args.provider, fallback_to_mock=True)

    # Instantiate orchestrator
    orchestrator = MultiAgentOrchestrator(
        retriever=retriever,
        provider=provider
    )

    gt_docs = args.gt_doc_ids.split(",") if args.gt_doc_ids else []

    request = VerificationRequest(
        claim_id=args.claim_id,
        claim_text=args.claim,
        dataset=args.dataset,
        ground_truth_label=args.ground_truth,
        ground_truth_doc_ids=gt_docs,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k
    )

    result = orchestrator.verify(request, pre_retrieved_evidence=fallback_evidence, run_id="cli_demo_run")

    logger.info("\n=== VERIFICATION RESULT ===")
    print(result.to_json())


if __name__ == "__main__":
    main()
