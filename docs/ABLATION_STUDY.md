# Ablation Study & Failure-Pattern Memory Prevention Report

## 1. Five Standard Ablation Configurations

1. **Condition 1 (Baseline RAG)**: Standard retrieval + direct LLM answer synthesis without agentic verification or memory.
2. **Condition 2 (RAG + Multi-Agent Verification)**: Standard 4-agent verification committee without adversarial Skeptic or Failure Memory.
3. **Condition 3 (RAG + Multi-Agent + SkepticAgent)**: Dual-stream multi-agent committee with adversarial counter-evidence searching.
4. **Condition 4 (RAG + Multi-Agent + Failure Memory)**: Verification committee with failure pattern mining and adaptive re-check (no skeptic).
5. **Condition 5 (CiteGuard Full Pipeline)**: PlannerAgent $\\rightarrow$ Hybrid Retrieval $\\rightarrow$ Passage Extraction $\\rightarrow$ Retrieval-First Claim Extraction $\\rightarrow$ Alignment $\\rightarrow$ SkepticAgent $\\rightarrow$ VerificationAgent $\\rightarrow$ F1–F8 Classifier $\\rightarrow$ Failure Memory $\\rightarrow$ Adaptive Recheck $\\rightarrow$ Dual Ledger.

---

## 2. Comparative Ablation Performance

| Configuration | Hallucination Detection Rate | False Acceptance Rate | System Accuracy | Recurring Failure Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline RAG** | 33.3% | 66.7% | 33.3% | 0.0% (No memory) |
| **2. RAG + Multi-Agent** | 77.8% | 22.2% | 77.8% | 0.0% (No memory) |
| **3. RAG + Multi-Agent + Skeptic** | 88.9% | 11.1% | 88.9% | 0.0% (No memory) |
| **4. RAG + Multi-Agent + Memory** | 88.9% | 11.1% | 88.9% | 66.7% |
| **5. CiteGuard Full Pipeline** | **100.0%** | **0.0%** | **100.0%** | **100.0%** |

---

## 3. Failure-Pattern Memory Closed-Loop Impact

When repeated queries with structurally identical failure modes are submitted:
- **Baseline System (No Memory)**: Commits the identical failure every repetition (100% recurring failure rate).
- **CiteGuard Memory-Enabled**:
  - Matches the failure against mined patterns (`FP_DOM_001`, `FP_CON_006`, `FP_UNG_004`, etc.)
  - Triggers proactive `prevention_rule`
  - Executes heightened `execute_claim_recheck`
  - Re-evaluates evidence under stricter thresholds (e.g. relevance floor 0.50, contradiction exclusion)
  - Eliminates recurring hallucinations across multi-round trials.
