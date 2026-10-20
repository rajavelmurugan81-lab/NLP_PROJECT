# CiteGuard Multi-Agent Verification Architecture & Documentation

## 1. Overview

Phase 6 implements the **Adaptive Multi-Agent Scientific Claim Verification Pipeline** for CiteGuard. The framework orchestrates specialized AI agents to evaluate scientific claims against candidate evidence retrieved via Phase 4's hybrid retrieval system (BM25 + Dense FAISS).

When agents disagree or evidence confidence is low, an adaptive verification policy triggers a comparative debate resolution process (`DebateResolver`). If a prediction disagrees with ground-truth labels during evaluation, Phase 5's `FailureAnalyzer` automatically generates a structured `FailureSignature` and persists it into the `HallucinationLedger`.

---

## 2. System Architecture

```
[VerificationRequest: Claim + Context]
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Phase 4 Hybrid Evidence        │
  │  Retriever (BM25 + FAISS)       │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │ Shared Evidence Context         │
  └────────────────┬────────────────┘
                   │
         ┌─────────┴─────────┐
         │ Parallel Agents   │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
┌─────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐
│Evidence │  │  Claim    │  │  Source  │  │Verification│
│ Agent   │  │ Analysis  │  │ Quality  │  │   Agent    │
└────┬────┘  └─────┬─────┘  └────┬─────┘  └─────┬──────┘
     │             │             │              │
     └─────────────┼─────────────┴──────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │ VerificationPolicy Evaluation   │
  │ (Check Agreement & Confidence)  │
  └────────────────┬────────────────┘
                   │
         ┌─────────┴─────────┐
       [YES]               [NO] (Disagreement or Low Confidence)
         │                   │
         │                   ▼
         │         ┌───────────────────┐
         │         │  DebateResolver   │
         │         │  (Comparative     │
         │         │   Synthesis)      │
         │         └─────────┬─────────┘
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │ Final Verdict & Provenance Trace│
  │ (SUPPORT / REFUTE / NEI)        │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │ Phase 5 Ledger Integration      │
  │ (Log Failure Signature if GT    │
  │  mismatch occurred)             │
  └─────────────────────────────────┘
```

---

## 3. Agent Roles & Specifications

| Agent Role | Class | System Role & Guidelines | Input / Output Contract |
| :--- | :--- | :--- | :--- |
| **Evidence Agent** | `EvidenceAgent` | Strictly evaluates retrieved evidence passages against the claim. Cites supporting or refuting document IDs. | `context` -> `AgentAssessment` (`decision`, `confidence`, `supporting_evidence_ids`, `contradicting_evidence_ids`, `rationale`) |
| **Claim Analysis Agent** | `ClaimAnalysisAgent` | Decomposes claim into atomic assertions and required evidence scope (in-vitro vs. clinical, correlation vs. causation). | `context` -> `AgentAssessment` (`decision`, `confidence`, `rationale`, `uncertainty`) |
| **Source Quality Agent** | `SourceQualityAgent` | Evaluates document metadata, title/abstract presence, and text completeness. Flags noisy or truncated passages. | `context` -> `AgentAssessment` (`decision`, `confidence`, `rationale`, `uncertainty`) |
| **Verification Agent** | `VerificationAgent` | Lead synthesis agent providing preliminary grounded verdict by balancing evidence, scope, and source quality. | `context` -> `AgentAssessment` (`decision`, `confidence`, `rationale`, `uncertainty`) |
| **Debate Resolver** | `DebateResolver` | Resolves multi-agent disagreement via comparative evidence synthesis. Prioritizes grounded evidence over simple voting. | `assessments` -> `DebateResult` (`rounds`, `revised_assessments`, `consensus_reached`, `final_decision`, `resolution_rationale`) |

---

## 4. LLM Provider Abstraction & Security

CiteGuard abstracts LLM provider interactions behind the `LLMProvider` base class:
- `MockProvider`: Provides deterministic, rule-driven responses for 100% offline unit testing without network or API dependencies.
- `OpenAIProvider`: Executes API requests to OpenAI models (`gpt-4o-mini`, `gpt-4o`) using environment variables.
- `get_provider(provider_type, fallback_to_mock=True)`: Provider factory with automatic fallback to `MockProvider` if no API key is detected.

### Security Guarantees:
- API keys are loaded strictly via `OPENAI_API_KEY` from environment variables (`.env`).
- `.env` is listed in `.gitignore` and never committed to source control.
- `.env.example` contains placeholder variables only.

---

## 5. Adaptive Verification Policy

The `VerificationPolicy` defines configurable routing rules:

```python
@dataclass
class VerificationPolicy:
    confidence_threshold: float = 0.70
    agreement_threshold: float = 0.75
    evidence_quality_threshold: float = 0.30
    max_reasoning_rounds: int = 3
    max_retrieved_evidence: int = 5
    fallback_behavior: str = "NEI"
```

- **Consensus Route**: If $\frac{\text{majority\_votes}}{\text{total\_agents}} \ge 0.75$ and mean confidence $\ge 0.70$, the pipeline outputs the majority verdict directly.
- **Debate Route**: If agreement falls below 0.75 or confidence is below 0.70, `DebateResolver` is invoked to perform comparative synthesis.
- **Insufficient Evidence Route**: If top evidence score $< 0.30$ or no evidence is retrieved, the pipeline returns `NEI` directly without invoking expensive LLM debate calls.

---

## 6. Phase 5 Ledger Integration & Traceability

When ground-truth labels are available in the `VerificationRequest`:
1. The orchestrator compares `final_label` against `ground_truth_label`.
2. If `final_label != ground_truth_label`, `FailureAnalyzer.analyze_instance(...)` classifies the failure mode (`RETRIEVAL_FAILURE`, `REASONING_FAILURE`, `CONTRADICTORY_EVIDENCE`, `INSUFFICIENT_EVIDENCE`, etc.).
3. The resulting `FailureSignature` is appended to `results/hallucination_ledger/hallucination_ledger.jsonl`.
4. Full execution metrics (`ExecutionMetrics`: execution time, token usage, agent call count, reasoning rounds) are captured in `VerificationResult`.

---

## 7. Configuration & CLI Usage

### Basic CLI Usage (Mock Mode):
```powershell
python scripts/verify_claim.py --claim "MicroRNA-21 regulates apoptosis in cancer cells" --retrieval-mode hybrid
```

### LLM Mode (OpenAI):
```powershell
# Set API Key in environment or .env
$env:OPENAI_API_KEY="your_api_key"

python scripts/verify_claim.py --claim "MicroRNA-21 regulates apoptosis in cancer cells" --provider openai
```

### Test Suite Execution:
```powershell
python -m pytest tests/verification/
```
