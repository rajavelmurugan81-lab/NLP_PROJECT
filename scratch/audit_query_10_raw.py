"""Script to run Query 10 and print the complete raw JSON result and inspection parameters."""

import os
import sys
import json
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.claim_generation.classifier import classify_question
from src.retrieval.paper_evidence import check_paper_domain_relevance

QUERY_10 = "what are the 10 most cited papers published in 2026 on quantum resistant federated learning and provide their exact DOI,citation count and experimental results?"

def main():
    print("=== QUERY 10 INTEGRITY AUDIT ===")
    
    # 1. Inspect classification & extraction
    q_class = classify_question(QUERY_10)
    print("\n--- 1. QUESTION CLASSIFICATION ---")
    print(f"question_type: {q_class['question_type'].value}")
    print(f"extracted topic: {q_class['topic']}")
    print(f"extracted target_year: {q_class['target_year']}")
    print(f"extracted target_count (N): {q_class['target_count']}")
    print(f"extracted ranking_metric: {q_class['ranking_metric']}")
    print(f"requires_doi: {q_class['requires_doi']}")
    print(f"requires_citations: {q_class['requires_citations']}")
    print(f"requires_experimental_results: {q_class['requires_experimental_results']}")

    # 2. Call live API endpoint POST /api/verify
    print("\n--- 2. CALLING LIVE API (POST /api/verify) ---")
    url = "http://127.0.0.1:8000/api/verify"
    req_body = json.dumps({
        "question": QUERY_10,
        "mode": "openalex_hybrid",
        "top_k": 5
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=req_body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))

    # 3. Print complete raw JSON result to file and summarize
    raw_path = "results/hallucination_ledger/query_10_complete_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)
    print(f"Saved complete raw JSON to: {raw_path}")

    print("\n--- 3. DETAILED VERIFICATION PARAMETERS ---")
    print(f"OpenAlex Status: {res_data.get('openalex_status')}")
    
    # Analyze papers
    claims = res_data.get("claims", [])
    all_evidence = []
    for c in claims:
        all_evidence.extend(c.get("evidence", []))

    openalex_count = sum(1 for e in all_evidence if "openalex" in str(e.get("retrieval_method", "")).lower())
    local_count = sum(1 for e in all_evidence if "openalex" not in str(e.get("retrieval_method", "")).lower())
    
    # Check domain validity
    domain_valid_count = 0
    for e in all_evidence:
        is_rel, sc = check_paper_domain_relevance(QUERY_10, e.get("passage", "") or e.get("text", ""), e.get("title", ""))
        if is_rel:
            domain_valid_count += 1

    print(f"Total Evidence Passages across claims: {len(all_evidence)}")
    print(f"Number of OpenAlex results in evidence: {openalex_count}")
    print(f"Number of locally retrieved results: {local_count}")
    print(f"Number of domain-valid results: {domain_valid_count}")

    print("\n--- 4. FINAL PAPERS, DOI, AND CITATION METRICS ---")
    seen_papers = set()
    for e in all_evidence:
        pid = e.get("paper_id") or e.get("doc_id")
        if pid not in seen_papers:
            seen_papers.add(pid)
            print(f"  * Paper ID: {pid}")
            print(f"    Title: {e.get('title')}")
            print(f"    DOI: {e.get('doi') or e.get('url_doi') or 'None'}")
            print(f"    Publication Year: {e.get('publication_year') or e.get('year')}")

    print("\n--- 5. RUNTIME AUDIT RECORDS ---")
    runtime_audits = res_data.get("runtime_audit", [])
    for a in runtime_audits:
        print(f"  * Claim: {a.get('claim_id')} -> {a.get('claim_text')}")
        print(f"    Verdict: {a.get('claim_verdict')} | Conf: {a.get('confidence')} | Audit Type: {a.get('failure_type')} | Severity: {a.get('severity')} | Status: {a.get('ledger_status')}")
        print(f"    Explanation: {a.get('explanation')}")

if __name__ == "__main__":
    main()
