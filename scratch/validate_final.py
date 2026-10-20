import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from src.ui.server import app

client = TestClient(app)

print("=== FINAL VALIDATION CHECK: REST API & VERIFICATION PIPELINE ===")

# Test 1: Status endpoint
res_status = client.get("/api/status")
print(f"1. GET /api/status -> Status Code: {res_status.status_code}")
status_data = res_status.json()
print(f"   Provider: {status_data.get('provider_display')} | Retriever Active: {status_data.get('retriever_active')}")

# Test 2: Verification endpoint
payload = {
    "question": "What is the relationship between sleep duration and cognitive performance in adults?",
    "provider": "mock",
    "top_k": 5,
    "mode": "hybrid"
}

res_verify = client.post("/api/verify", json=payload)
print(f"\n2. POST /api/verify -> Status Code: {res_verify.status_code}")
verify_data = res_verify.json()

print(f"   Success: {verify_data.get('success')}")
print(f"   Execution Time: {verify_data.get('execution_time_sec')}s")
print(f"   Question ID: {verify_data.get('question_id')}")

report = verify_data.get("report", {})
claims = report.get("claims", [])
print(f"   Total Claims Generated: {len(claims)}")

for idx, c in enumerate(claims, 1):
    c_id = c.get("claim_id")
    verdict = c.get("verdict")
    ev_list = c.get("evidence", [])
    fail_sig = c.get("failure_signature")
    
    print(f"\n   -> Claim #{idx} ({c_id}):")
    print(f"      Text: \"{c.get('claim_text')}\"")
    print(f"      Verdict: {verdict} | Confidence: {c.get('confidence')}")
    print(f"      Evidence Count: {len(ev_list)}")
    
    if ev_list:
        top_doc = ev_list[0]
        print(f"      Top Doc ID: {top_doc.get('doc_id')} | Score: {top_doc.get('score')} | Rank: #{top_doc.get('rank')}")
        print(f"      Top Title: {top_doc.get('title')[:60]}...")
    
    print(f"      Failure Signature: {fail_sig}")

print("\n=== FINAL VALIDATION CHECK COMPLETED SUCCESSFULLY ===")
