import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import websockets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
WORKSPACE_DIR = r"c:\Users\rajav\OneDrive\Desktop\nlp\nlp"
DOCS_SCREENSHOTS_DIR = os.path.join(WORKSPACE_DIR, "docs", "screenshots")
ARTIFACT_DIR = r"C:\Users\rajav\.gemini\antigravity\brain\50425fa8-edf7-42d8-91ab-bbf4611931bc"
USER_DATA = os.path.join(WORKSPACE_DIR, "chrome_walkthrough_profile")
PORT = 9223

os.makedirs(DOCS_SCREENSHOTS_DIR, exist_ok=True)

def start_chrome():
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={USER_DATA}",
        "--window-size=1440,960",
        "--hide-scrollbars",
        "about:blank"
    ]
    proc = subprocess.Popen(cmd)
    time.sleep(2.5)
    return proc

async def run_capture():
    resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")
    targets = json.loads(resp.read().decode())
    
    # Filter for standard page target
    page_target = next((t for t in targets if t.get("type") == "page"), targets[0])
    ws_url = page_target["webSocketDebuggerUrl"]
    print(f"Connected to page target '{page_target.get('title')}' at {ws_url}", flush=True)

    async with websockets.connect(ws_url, max_size=30*1024*1024) as ws:
        msg_id = 1

        async def call(method, params=None):
            nonlocal msg_id
            m_id = msg_id
            msg_id += 1
            payload = {"id": m_id, "method": method, "params": params or {}}
            await ws.send(json.dumps(payload))
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("id") == m_id:
                    return data.get("result", {})

        async def eval_js(expr):
            return await call("Runtime.evaluate", {"expression": expr, "awaitPromise": True})

        async def capture(step_num, filename, description):
            res = await call("Page.captureScreenshot", {"format": "png"})
            img_bytes = base64.b64decode(res["data"])
            
            dest_local = os.path.join(DOCS_SCREENSHOTS_DIR, filename)
            with open(dest_local, "wb") as f:
                f.write(img_bytes)
                
            dest_artifact = os.path.join(ARTIFACT_DIR, filename)
            with open(dest_artifact, "wb") as f:
                f.write(img_bytes)
                
            print(f"[{step_num:02d}/15] Saved: {filename} ({len(img_bytes)} bytes) - {description}", flush=True)

        await call("Page.enable")
        await call("Runtime.enable")

        # ----------------------------------------------------
        # 1. Landing Page / Research Workspace Initial View
        # ----------------------------------------------------
        print("1. Navigating to CiteGuard application...", flush=True)
        await call("Page.navigate", {"url": "http://127.0.0.1:8000"})
        await asyncio.sleep(2.5)
        await capture(1, "01_research_workspace_initial.png", "Research Workspace landing view with input form and sidebar navigation")

        # ----------------------------------------------------
        # 2. Query Selection & Configuration
        # ----------------------------------------------------
        print("2. Populating research question and options...", flush=True)
        await eval_js("""
            (() => {
                const input = document.getElementById('questionInput');
                input.value = "How does model size affect the accuracy and computational cost of large language models?";
                const mode = document.getElementById('modeSelect');
                if (mode) mode.value = "openalex_hybrid";
                const topk = document.getElementById('topKSelect');
                if (topk) topk.value = "5";
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(2, "02_query_selection_and_parameters.png", "Configured research query and hybrid retrieval mode parameters")

        # ----------------------------------------------------
        # 3. Trigger Verification & Capture Active Stepper
        # ----------------------------------------------------
        print("3. Submitting verification request to trigger 8-stage stepper...", flush=True)
        await eval_js("""
            (() => {
                const btn = document.getElementById('submitBtn');
                btn.click();
            })()
        """)
        await asyncio.sleep(0.9)
        await capture(3, "03_stepper_pipeline_execution.png", "Active 8-stage pipeline stepper showing live multi-agent verification progress")

        # ----------------------------------------------------
        # Wait for Verification to Complete
        # ----------------------------------------------------
        print("Waiting for verification results to render...", flush=True)
        for i in range(50):
            res = await eval_js("""
                (() => {
                    const c = document.getElementById('resultsContainer');
                    return c && !c.classList.contains('hidden') && c.style.display !== 'none';
                })()
            """)
            if res.get("result", {}).get("value") is True:
                print(f"Results rendered successfully after ~{i*0.5:.1f}s!", flush=True)
                break
            await asyncio.sleep(0.5)

        await asyncio.sleep(1.5)

        # ----------------------------------------------------
        # 4. Results Overview & Metrics Dashboard
        # ----------------------------------------------------
        await eval_js("window.scrollTo({top: 0, behavior: 'instant'});")
        await asyncio.sleep(0.5)
        await capture(4, "04_results_overview_and_metrics.png", "Results overview banner showing latency, confidence score, and verdict distribution")

        # ----------------------------------------------------
        # 5. Multi-Agent Research Plan
        # ----------------------------------------------------
        await eval_js("""
            (() => {
                const el = document.getElementById('retrievalSummaryBox') || document.querySelector('.retrieval-summary-card') || document.querySelector('.plan-box');
                if (el) el.scrollIntoView({behavior: 'instant', block: 'center'});
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(5, "05_multi_agent_research_plan.png", "Multi-agent research plan showing sub-claim hypotheses and literature retrieval strategy")

        # ----------------------------------------------------
        # 6. Verified Claims & Verdicts List
        # ----------------------------------------------------
        await eval_js("""
            (() => {
                const el = document.getElementById('claimsList');
                if (el) el.scrollIntoView({behavior: 'instant', block: 'start'});
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(6, "06_verified_claims_verdict_list.png", "Decomposed atomic claims showing verdict status badges and numerical confidence")

        # ----------------------------------------------------
        # 7. Detailed Evidence Passages & Provenance
        # ----------------------------------------------------
        await eval_js("""
            (() => {
                const btn = document.querySelector('.btn-toggle-evidence') || document.querySelector('.accordion-toggle');
                if (btn) btn.click();
                const list = document.getElementById('verificationDetailsList') || document.querySelector('.evidence-item') || document.querySelector('.claim-card');
                if (list) list.scrollIntoView({behavior: 'instant', block: 'start'});
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(7, "07_evidence_passages_and_provenance.png", "Expanded evidence cards showing cited scientific papers, publication metadata, and DOI links")

        # ----------------------------------------------------
        # 8. Adaptive Failure Re-check & Pattern Mitigation
        # ----------------------------------------------------
        await eval_js("""
            (() => {
                const card = document.querySelector('.claim-card:nth-child(2)') || document.querySelector('.claim-card');
                if (card) card.scrollIntoView({behavior: 'instant', block: 'center'});
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(8, "08_failure_pattern_adaptive_recheck.png", "Verification card displaying reasoning, evidence grounding, and adaptive pattern checks")

        # ----------------------------------------------------
        # 9. Academic Report Synthesis & Verdict Dashboard
        # ----------------------------------------------------
        await eval_js("""
            (() => {
                const el = document.getElementById('synthesisVerdictSummary') || document.querySelector('.synthesis-container');
                if (el) el.scrollIntoView({behavior: 'instant', block: 'start'});
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(9, "09_academic_report_synthesis.png", "Academic final report executive synthesis and verdict distribution dashboard")

        # ----------------------------------------------------
        # 10. Uncertainty, Research Gaps & Export Actions
        # ----------------------------------------------------
        await eval_js("""
            (() => {
                const el = document.getElementById('limitationsList') || document.getElementById('downloadJsonBtn') || document.querySelector('.report-footer');
                if (el) el.scrollIntoView({behavior: 'instant', block: 'center'});
            })()
        """)
        await asyncio.sleep(0.5)
        await capture(10, "10_academic_report_gaps_and_export.png", "Scientific epistemic limitations, contradictory evidence, and JSON/Markdown export controls")

        # ----------------------------------------------------
        # 11. Trusted Memory Ledger View
        # ----------------------------------------------------
        print("11. Navigating to Trusted Memory view...", flush=True)
        await eval_js("""
            (() => {
                const btn = document.getElementById('navTrustedBtn');
                if (btn) btn.click();
            })()
        """)
        await asyncio.sleep(1.2)
        await capture(11, "11_trusted_memory_ledger.png", "Trusted Memory view displaying 261 admitted scientific claims passing the 7-rule policy")

        # ----------------------------------------------------
        # 12. Failure-Pattern Memory View
        # ----------------------------------------------------
        print("12. Navigating to Failure-Pattern Memory view...", flush=True)
        await eval_js("""
            (() => {
                const btn = document.getElementById('navPatternsBtn');
                if (btn) btn.click();
            })()
        """)
        await asyncio.sleep(1.2)
        await capture(12, "12_failure_pattern_memory.png", "Failure-Pattern Memory view displaying 21 persisted hallucination patterns (FP_DOM_001-FP_SQF_021)")

        # ----------------------------------------------------
        # 13. Runtime Audit Ledger View
        # ----------------------------------------------------
        print("13. Navigating to Runtime Audit Ledger view...", flush=True)
        await eval_js("""
            (() => {
                const btn = document.getElementById('navAuditBtn');
                if (btn) btn.click();
            })()
        """)
        await asyncio.sleep(1.2)
        await capture(13, "13_runtime_audit_ledger.png", "Runtime Audit Ledger displaying live verification audit logs, execution latencies, and admission events")

        # ----------------------------------------------------
        # 14. Historical Benchmark Ledger View
        # ----------------------------------------------------
        print("14. Navigating to Historical Benchmark Ledger view...", flush=True)
        await eval_js("""
            (() => {
                const btn = document.getElementById('navBenchmarkBtn');
                if (btn) btn.click();
            })()
        """)
        await asyncio.sleep(1.2)
        await capture(14, "14_historical_benchmark_ledger.png", "Historical Benchmark Ledger view showing SciFact/SciClaimHunt ground-truth evaluation records")

        # ----------------------------------------------------
        # 15. System Architecture & Health Modal
        # ----------------------------------------------------
        print("15. Opening System Architecture modal...", flush=True)
        await eval_js("""
            (() => {
                const btn = document.getElementById('navSystemBtn');
                if (btn) btn.click();
            })()
        """)
        await asyncio.sleep(1.0)
        await capture(15, "15_system_architecture_modal.png", "System Architecture modal detailing pipeline components, hybrid retrieval, and memory safeguards")

        print("=== Complete! All 15 screenshots captured and saved to docs/screenshots and artifacts. ===", flush=True)

if __name__ == "__main__":
    proc = start_chrome()
    try:
        asyncio.run(run_capture())
    finally:
        proc.terminate()
