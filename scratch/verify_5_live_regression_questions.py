"""Script to verify CiteGuard end-to-end against the 5 mandatory live regression questions."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.research_pipeline import ResearchPipeline

QUESTIONS = [
    "Does sleep improve cognitive performance?",
    "How does increasing model size affect LLM performance?",
    "Does miRNA regulation influence cancer progression?",
    "How does CRISPR improve gene editing efficiency?",
    "what are the 10 most cited papers published in 2026 on quantum resistant federated learning and provide their exact DOI,citation count and experimental results?"
]

def main():
    print("=" * 70)
    print("STARTING 5 LIVE REGRESSION QUERIES AUDIT")
    print("=" * 70)

    pipeline = ResearchPipeline(provider_type="mock")
    all_results = []

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"\n[{idx}/5] Evaluating: \"{q}\"")
        t0 = time.time()
        res = pipeline.run_pipeline(
            research_question=q,
            top_k=5,
            mode="openalex_hybrid",
            verbose=False
        )
        elapsed = time.time() - t0
        rq = res["research_question"]
        q_type = res["question_type"]
        oa_status = res["openalex_status"]
        audit_records = res.get("runtime_audit_records", [])

        # Inspect generated claims
        claims_summary = []
        for c, v, a in zip(rq.generated_claims, res["verification_results"], audit_records):
            claims_summary.append({
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "verdict": v.final_label,
                "confidence": v.confidence,
                "evidence_count": len(v.evidence),
                "failure_type": a.get("failure_type"),
                "severity": a.get("severity"),
                "ledger_status": a.get("ledger_status"),
                "explanation": a.get("explanation")
            })

        print(f" -> Question Type: {q_type}")
        print(f" -> OpenAlex Status: {oa_status}")
        print(f" -> Papers Retrieved: {len(res['papers_retrieved'])}")
        print(f" -> Claims Verified: {len(claims_summary)}")
        for cs in claims_summary:
            print(f"    * [{cs['claim_id']}] \"{cs['claim_text']}\" -> Verdict: {cs['verdict']} (Conf: {cs['confidence']:.2f}) | Audit: {cs['failure_type']} ({cs['ledger_status']})")

        all_results.append({
            "index": idx,
            "question": q,
            "question_type": q_type,
            "openalex_status": oa_status,
            "elapsed_sec": round(elapsed, 2),
            "claims": claims_summary
        })

    out_path = "results/hallucination_ledger/five_queries_audit_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE. Report written to {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
