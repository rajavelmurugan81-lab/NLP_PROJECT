"""Main CLI Entry Point for CiteGuard Scientific Claim Verification System."""

import argparse
import json
import os
import sys
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CiteGuardPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("citeguard_cli")


def main():
    parser = argparse.ArgumentParser(description="CiteGuard Scientific Claim Verification System CLI")
    parser.add_argument("--claim", type=str, required=True, help="Scientific claim text to verify")
    parser.add_argument("--claim-id", type=str, default="cli_claim_001", help="Claim Identifier")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset tag")
    parser.add_argument("--split", type=str, default="test", help="Dataset split tag")
    parser.add_argument("--mode", "--retrieval-mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"], help="Retrieval mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence items to retrieve")
    parser.add_argument("--provider", type=str, default="mock", help="LLM Provider type ('mock' or 'openai')")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional ground-truth label")
    parser.add_argument("--gt-doc-ids", type=str, default=None, help="Comma-separated list of ground-truth doc IDs")
    parser.add_argument("--output", type=str, default=None, help="Path to save result JSON file")
    parser.add_argument("--pretty", action="store_true", help="Format JSON output with indentation")
    args = parser.parse_args()

    logger.info("=== CiteGuard Scientific Claim Verification System ===")
    logger.info(f"Submitted Claim: \"{args.claim}\"")
    logger.info(f"Retrieval Mode: {args.mode} | LLM Provider: {args.provider}")

    # Instantiate integrated pipeline
    pipeline = CiteGuardPipeline(provider_type=args.provider)

    gt_docs = args.gt_doc_ids.split(",") if args.gt_doc_ids else []

    # Execute verification
    result = pipeline.verify(
        claim=args.claim,
        claim_id=args.claim_id,
        dataset=args.dataset,
        split=args.split,
        ground_truth_label=args.ground_truth,
        ground_truth_doc_ids=gt_docs,
        top_k=args.top_k,
        mode=args.mode,
        run_id="cli_run"
    )

    res_dict = result.to_dict()
    output_json = json.dumps(res_dict, indent=2 if args.pretty else None)

    # Output result to console
    logger.info("\n=== VERIFICATION RESULT ===")
    print(output_json)

    # Optionally save result to file
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json.dumps(res_dict, indent=2))
        logger.info(f"Verification result saved to: {args.output}")

    if result.final_label == "ERROR":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
