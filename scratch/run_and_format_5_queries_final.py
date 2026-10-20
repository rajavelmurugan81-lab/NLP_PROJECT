"""Script to run all 5 live regression queries, format detailed outputs, and save complete outputs."""

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
    print("=" * 80)
    print("EXECUTING ALL 5 LIVE REGRESSION QUERIES FOR FINAL AUDIT")
    print("=" * 80)

    pipeline = ResearchPipeline(provider_type="mock")
    final_outputs = []

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"\n[{idx}/5] RUNNING: \"{q}\"")
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
        papers = res.get("papers_retrieved", [])
        audit_records = res.get("runtime_audit_records", [])
        ver_results = res.get("verification_results", [])

        # Format claims and verification
        claims_detail = []
        for c, v, a in zip(rq.generated_claims, ver_results, audit_records):
            ev_list = []
            for ev in v.evidence:
                ev_list.append({
                    "doc_id": ev.doc_id,
                    "title": ev.title,
                    "score": ev.score,
                    "doi": getattr(ev, "metadata", {}).get("doi", "") or getattr(ev, "metadata", {}).get("url_doi", ""),
                    "source": getattr(ev, "source", "openalex")
                })

            claims_detail.append({
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "verdict": v.final_label,
                "confidence": v.confidence,
                "evidence_count": len(v.evidence),
                "audit": {
                    "failure_type": a.get("failure_type"),
                    "failure_signature": a.get("failure_signature"),
                    "severity": a.get("severity"),
                    "ledger_status": a.get("ledger_status"),
                    "explanation": a.get("explanation")
                },
                "evidence_sample": ev_list[:2]
            })

        q_summary = {
            "query_number": idx,
            "query_text": q,
            "question_type": q_type,
            "openalex_status": oa_status,
            "papers_retrieved_count": len(papers),
            "execution_time_sec": round(elapsed, 2),
            "claims": claims_detail
        }
        final_outputs.append(q_summary)

        print(f"  -> Type: {q_type} | OpenAlex: {oa_status} | Time: {elapsed:.2f}s")
        for cd in claims_detail:
            print(f"    * [{cd['claim_id']}] \"{cd['claim_text']}\"")
            print(f"      Verdict: {cd['verdict']} (Conf: {cd['confidence']}) | Audit: {cd['audit']['failure_type']} ({cd['audit']['ledger_status']})")

    report_path = "results/hallucination_ledger/final_5_queries_audit_complete.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_outputs, f, indent=2)
    print("\n" + "=" * 80)
    print(f"SAVED FINAL 5 QUERIES COMPLETE AUDIT REPORT TO: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
