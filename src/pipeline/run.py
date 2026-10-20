"""CLI Entry Point for Phase 10: Research Question to Verified Scientific Report Pipeline."""

import os
import sys
import json
import argparse
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.research_pipeline import ResearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("src.pipeline.run")


def main():
    parser = argparse.ArgumentParser(
        description="CiteGuard: Research Question to Automatically Generated Claims to Verified Scientific Report Pipeline"
    )
    parser.add_argument(
        "--question", "-q", type=str, required=True,
        help="User research question (e.g. 'What is the relationship between sleep duration and cognitive performance?')"
    )
    parser.add_argument(
        "--provider", "-p", type=str, default="mock",
        choices=["mock", "openai", "anthropic", "ollama", "custom"],
        help="LLM provider for claim generation and multi-agent verification (default: 'mock')"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="hybrid",
        choices=["hybrid", "sparse", "dense", "none"],
        help="Evidence retrieval mode (default: 'hybrid')"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5,
        help="Number of evidence passages to retrieve per claim (default: 5)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="results/reports",
        help="Directory to save JSON and Markdown reports (default: 'results/reports')"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Print pretty-formatted JSON summary output to stdout"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose progress print statements"
    )

    args = parser.parse_args()

    pipeline = ResearchPipeline(
        provider_type=args.provider,
        output_dir=args.output_dir
    )

    res = pipeline.run_pipeline(
        research_question=args.question,
        top_k=args.top_k,
        mode=args.mode,
        verbose=not args.quiet
    )

    if args.pretty:
        report_json_path = res["report_files"]["json_path"]
        if os.path.exists(report_json_path):
            with open(report_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("\n=== FINAL GENERATED REPORT (JSON SUMMARY) ===")
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
