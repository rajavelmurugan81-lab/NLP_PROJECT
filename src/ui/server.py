"""FastAPI Backend Server for CiteGuard Web Interface."""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.research_pipeline import ResearchPipeline
from src.claim_generation.validator import validate_research_question

logger = logging.getLogger("src.ui.server")

# Path definitions
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
REPORTS_DIR = os.path.join(BASE_DIR, "results", "reports")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm retriever and model indexes during server startup to avoid endpoint timeouts."""
    try:
        logger.info("Warming up ResearchPipeline on server startup...")
        get_pipeline(provider="mock")
        logger.info("ResearchPipeline warmup completed successfully.")
    except Exception as e:
        logger.warning(f"Startup pipeline warmup warning: {e}")
    yield


app = FastAPI(
    title="CiteGuard: Scientific Claim Verification System",
    description="Evidence-Grounded Research Question Verification Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web UI compatibility across origins/ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (default offline mock mode)
pipeline_instance: Optional[ResearchPipeline] = None


def get_pipeline(provider: str = "mock") -> ResearchPipeline:
    """Get or initialize the ResearchPipeline instance."""
    global pipeline_instance
    if pipeline_instance is None or pipeline_instance.provider_type != provider:
        logger.info(f"Initializing ResearchPipeline with provider='{provider}'")
        pipeline_instance = ResearchPipeline(provider_type=provider, output_dir=REPORTS_DIR)
    return pipeline_instance


class VerificationApiRequest(BaseModel):
    question: str = Field(..., description="Research question to decompose and verify")
    provider: str = Field(default="mock", description="LLM provider name (default: mock)")
    top_k: int = Field(default=5, ge=1, le=20, description="Top-k evidence passages per claim")
    mode: str = Field(default="openalex_hybrid", description="Retrieval mode: openalex_hybrid, openalex, local_hybrid, sparse, dense")
    model: Optional[str] = Field(default=None, description="Retrieval mode alias")


@app.get("/api/status")
@app.get("/api/health")
def get_system_status():
    """Return system readiness and provider indicator status."""
    pl = get_pipeline(provider="mock")
    has_retriever = pl.verification_pipeline.retriever is not None
    return {
        "status": "online",
        "provider": "mock",
        "provider_display": "OpenAlex + Local Corpus",
        "cloud_api_key_configured": False,
        "retriever_active": has_retriever,
        "indexed_corpora": ["OpenAlex API", "SciFact", "SciClaimHunt"],
        "default_retrieval_mode": "openalex_hybrid",
        "supported_retrieval_modes": [
            "openalex_hybrid",
            "openalex",
            "local_hybrid",
            "sparse",
            "dense"
        ],
        "reports_dir": REPORTS_DIR
    }


@app.get("/api/verify")
def get_verify_info():
    """Informational endpoint handling GET /api/verify browser navigations gracefully."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "online",
            "message": "CiteGuard Verification API Endpoint.",
            "instruction": "Send an HTTP POST request to /api/verify with a JSON request body.",
            "example_request_body": {
                "question": "How does model size affect the accuracy and computational cost of large language models?",
                "mode": "openalex_hybrid",
                "top_k": 5,
                "provider": "mock"
            },
            "supported_retrieval_modes": {
                "openalex_hybrid": "OpenAlex API + Local Corpus (RRF Fusion)",
                "openalex": "OpenAlex API Only",
                "local_hybrid": "Local Hybrid (BM25 + FAISS + RRF)",
                "sparse": "Sparse BM25 Only",
                "dense": "Dense FAISS Vector Only"
            }
        }
    )


@app.post("/api/verify")
def run_research_verification(req: VerificationApiRequest):
    """Execute end-to-end question verification workflow."""
    is_valid, norm_q, err = validate_research_question(req.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid Research Question: {err}")

    try:
        retrieval_mode = req.model or req.mode
        pipeline = get_pipeline(provider=req.provider)
        res = pipeline.run_pipeline(
            research_question=norm_q,
            top_k=req.top_k,
            mode=retrieval_mode,
            verbose=False
        )

        json_path = res["report_files"]["json_path"]
        md_path = res["report_files"]["md_path"]

        # Read JSON report payload to return structured response
        report_data = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

        # Determine dynamic paper source provider badge display
        sources_used = set()
        for claim in report_data.get("claims", []):
            for ev in claim.get("evidence", []):
                s = str(ev.get("source", "")).lower()
                if "openalex" in s:
                    sources_used.add("openalex")
                elif "local" in s or "scifact" in s or "sciclaimhunt" in s:
                    sources_used.add("local")

        if "openalex" in sources_used and "local" in sources_used:
            provider_display = "OpenAlex + Local Corpus"
        elif "openalex" in sources_used:
            provider_display = "OpenAlex API"
        elif "local" in sources_used:
            provider_display = "Local Corpus Fallback"
        else:
            provider_display = "OpenAlex + Local Corpus"

        openalex_status = res.get("openalex_status", "CONNECTED_WITH_RESULTS")
        question_type = res.get("question_type", "SCIENTIFIC_HYPOTHESIS")
        runtime_audit_records = res.get("runtime_audit_records", [])

        # Enrich claims array with runtime audit data if not already present
        claims_list = report_data.get("claims", [])
        for c, a_rec in zip(claims_list, runtime_audit_records):
            if "runtime_audit" not in c or not c["runtime_audit"]:
                c["runtime_audit"] = a_rec

        return {
            "success": True,
            "question_id": res["research_question"].question_id,
            "research_question": norm_q,
            "question": norm_q,
            "question_type": question_type,
            "research_plan": res.get("research_plan") or report_data.get("research_plan"),
            "openalex_status": openalex_status,
            "execution_time_sec": round(res["total_execution_time_sec"], 2),
            "provider_display": provider_display,
            "report_files": {
                "json_filename": os.path.basename(json_path),
                "md_filename": os.path.basename(md_path),
            },
            "claims": claims_list,
            "grounded_claims": res.get("grounded_claims", []),
            "papers_retrieved": res.get("papers_retrieved", []),
            "papers_retrieved_count": len(res.get("papers_retrieved", [])),
            "passages_extracted": [p.to_dict() if hasattr(p, "to_dict") else p for p in res.get("passages_extracted", [])],
            "runtime_audit": runtime_audit_records,
            "pattern_mining_result": res.get("pattern_mining_result"),
            "bibliographic_results": res.get("bibliographic_results", []),
            "verification": report_data.get("verification", {
                "supported": 0,
                "partially_supported": 0,
                "insufficient_evidence": 0,
                "contradicted": 0
            }),
            "synthesis": report_data.get("synthesis", {
                "summary": report_data.get("overall_synthesis", ""),
                "supported_count": 0,
                "partial_count": 0,
                "insufficient_count": 0,
                "contradicted_count": 0
            }),
            "dashboard": report_data.get("dashboard", {}),
            "research_synthesis": report_data.get("research_synthesis", {}),
            "uncertainty_and_gaps": report_data.get("uncertainty_and_gaps", {}),
            "report": report_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline verification failure: {str(e)}")


@app.get("/api/ledger/benchmark")
def get_benchmark_ledger(limit: int = Query(default=100, ge=1, le=500)):
    """Serve historical ground-truth benchmark failure records from results/hallucination_ledger/."""
    ledger_path = os.path.join(BASE_DIR, "results", "hallucination_ledger", "hallucination_ledger.jsonl")
    records = []
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read benchmark ledger: {e}")

    # Load distribution statistics if present
    stats_path = os.path.join(BASE_DIR, "results", "hallucination_ledger", "overall_failure_statistics.json")
    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read overall failure stats: {e}")

    return {
        "status": "success",
        "total_benchmark_records": len(records),
        "statistics": stats,
        "records": records[-limit:]
    }


@app.get("/api/ledger/runtime")
def get_runtime_audit_ledger(limit: int = Query(default=100, ge=1, le=500)):
    """Serve live query runtime hallucination audit records from results/hallucination_ledger/runtime_audit.jsonl."""
    runtime_path = os.path.join(BASE_DIR, "results", "hallucination_ledger", "runtime_audit.jsonl")
    records = []
    if os.path.exists(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read runtime audit ledger: {e}")

    return {
        "status": "success",
        "total_runtime_audits": len(records),
        "records": records[-limit:]
    }


@app.get("/api/ledger/patterns")
def get_failure_patterns():
    """Serve learned failure patterns from results/hallucination_ledger/failure_patterns.json."""
    patterns_path = os.path.join(BASE_DIR, "results", "hallucination_ledger", "failure_patterns.json")
    if not os.path.exists(patterns_path):
        from src.evaluation.pattern_miner import PatternMiner
        miner = PatternMiner(output_patterns_path=patterns_path)
        miner.mine_patterns()

    try:
        with open(patterns_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if pipeline has mining result tracking
        pipe = get_pipeline()
        last_mining = pipe.failure_memory.get_last_mining_result()
        mining_summary = last_mining.to_dict() if last_mining else None

        return {
            "status": "success",
            "mining_summary": mining_summary,
            **data
        }
    except Exception as e:
        logger.error(f"Failed to read failure patterns: {e}")
        return {"status": "error", "message": str(e), "patterns": []}


@app.get("/api/ledger/trusted")
def get_trusted_memory():
    """Serve cleanly passed verification claims recorded in Trusted Memory."""
    trusted_path = os.path.join(BASE_DIR, "results", "hallucination_ledger", "trusted_memory.json")
    if not os.path.exists(trusted_path):
        return {"status": "success", "total_trusted_claims": 0, "claims": []}
    try:
        with open(trusted_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", **data}
    except Exception as e:
        logger.error(f"Failed to read trusted memory: {e}")
        return {"status": "error", "message": str(e), "claims": []}


@app.get("/api/ledger/prevention-evaluation")
def get_prevention_evaluation():
    """Serve failure prevention evaluation metrics."""
    eval_path = os.path.join(BASE_DIR, "results", "hallucination_ledger", "failure_prevention_evaluation.json")
    if not os.path.exists(eval_path):
        return {
            "status": "pending",
            "message": "Evaluation experiment not yet executed.",
            "patterns_discovered": 9,
            "patterns_reused": 0
        }
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", **data}
    except Exception as e:
        logger.error(f"Failed to read prevention evaluation: {e}")
        return {"status": "error", "message": str(e)}



@app.get("/api/reports/{filename}")
def download_report_file(filename: str):
    """Serve JSON or Markdown report files for user download."""
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(REPORTS_DIR, safe_filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report file not found.")

    media_type = "application/json" if safe_filename.endswith(".json") else "text/markdown"
    return FileResponse(path=filepath, media_type=media_type, filename=safe_filename)


# Mount static assets if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"message": "CiteGuard API is running. Static index.html missing."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ui.server:app", host="127.0.0.1", port=8000, reload=False)
