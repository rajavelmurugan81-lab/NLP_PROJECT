# CiteGuard: Project Specification & Architecture

## 1. Executive Summary & Project Identification

* **Project Title**: CiteGuard: Failure-Aware Adaptive Verification & Closed-Loop Hallucination Memory for Scientific Claim Verification
* **System Name**: CiteGuard
* **Domain**: Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), Automated Fact-Checking, Scientific Claim Verification & LLM Hallucination Mitigation

---

## 2. Problem Statement & Research Gap

### 2.1 Problem Statement
Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks are increasingly deployed for scientific literature review, automated fact-checking, and evidence-based decision-making. However, existing systems suffer from critical failure modes:
1. **RAG & Context Bottlenecks**: Retrieving irrelevant or incomplete evidence, context conflicts, and query ambiguity lead to false claims or hallucinated justifications.
2. **Stateless Verification**: Existing claim verifiers evaluate each claim in isolation. They answer *"Is this claim supported or refuted?"* but cannot answer *"Have we made this structural reasoning mistake before?"*
3. **Uncalibrated Verification Depth**: Standard systems apply static, uniform verification pipelines regardless of claim complexity, evidence scope mismatch, or historical error patterns, leading to inefficiency or high false negative rates.

### 2.2 Key Research Gaps Identified
* **No Structured Failure Memory**: Lack of explicit mechanisms to store, index, and query structural reasoning failures across verification sessions.
* **Lack of Failure Signatures**: Current systems flag errors as binary outputs (`True`/`False`) without categorizing *why* the failure occurred (e.g., Overgeneralization, Scope Mismatch, Outdated Evidence).
* **Missing Closed-Loop Evolution**: Existing multi-agent debate and RAG pipelines are stateless and do not dynamically alter their verification strategies based on past failure signatures.

---

## 3. Project Objectives

1. **Structured Failure Representation**: Formulate a 8-category taxonomy of verification and hallucination failure types and implement a standardized **Failure Signature** structure.
2. **Persistent Hallucination Ledger**: Build a vector-indexed, queryable **Hallucination Ledger** that logs failure signatures and contextual claim features.
3. **Historical Failure Matching**: Implement a retrieval-based pre-verification step to search the Ledger for historical failure patterns before running deep verification.
4. **Failure-Aware Adaptive Verification**: Design a dynamic 3-tiered verification architecture (Level 1: Lightweight NLI -> Level 2: Multi-Agent Debate [Prosecutor / Skeptic / Leader] -> Level 3: Cross-Model & Scope/Contradiction Audit) triggered by calculated risk levels.
5. **Fine-Grained Claim-to-Source Provenance**: Provide sentence-level attribution linking atomic claims directly to explicit scientific evidence.
6. **Closed-Loop Evaluation**: Evaluate the framework on scientific benchmarks and measure the reduction in **Repeated Hallucination Rate (RHR)** across sequential evaluation rounds.

---

## 4. Proposed System Architecture (CiteGuard)

CiteGuard operates in a **6-Stage Closed Loop**:

```
[Input Scientific Claim & Context]
                │
                ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Atomic Claim Decomposition         │
│  & Hybrid Evidence Retrieval (BM25 + FAISS)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Historical Failure Matching        │
│  (Querying Hallucination Ledger)             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Risk Assessment & Adaptive Routing │
│  - Low Risk: Level 1 (Lightweight NLI)       │
│  - Med Risk: Level 2 (Multi-Agent Debate)    │
│  - High Risk: Level 3 (Cross-Model Audit)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 4: Claim Verification & Provenance    │
│  (Support / Refute / Partial / Neutral)      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 5: Failure Classification             │
│  & Failure Signature Generation              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 6: Closed-Loop Memory Update          │
│  (Write to Hallucination Ledger for Evolution)│
└──────────────────────────────────────────────┘
```

---

## 5. Failure Signature Taxonomy (8 Categories)

| ID | Failure Category | Description & Example Signature |
|:---|:---|:---|
| **F1** | **Unsupported Claim** | Claim asserts a fact not present in or backed by any retrieved evidence. |
| **F2** | **Citation Mismatch** | A paper/source is cited, but the cited sentence does not contain the claimed assertion. |
| **F3** | **Overgeneralization** | Claim makes broad universal statements based on narrow or conditional study samples. |
| **F4** | **Causal Leap** | Claim asserts a definitive causal relationship when evidence only shows correlation. |
| **F5** | **Outdated Evidence** | Claim relies on superseded scientific findings or obsolete metrics. |
| **F6** | **Context / Population Mismatch** | Claim applies findings from one domain/population (e.g., in-vitro) to another (e.g., human trials). |
| **F7** | **Contradictory Evidence** | System selects single supporting snippet while ignoring explicit contradictory evidence in the same paper. |
| **F8** | **Evidence Misinterpretation** | Model misinterprets statistical significance, p-values, or negative findings. |

---

## 6. Input / Output Specifications

### 6.1 Expected Input
* `claim_text` (str): Scientific assertion to verify.
* `context_documents` (optional list/dict): Full-text scientific papers, abstracts, or candidate PMIDs.
* `query_metadata` (optional dict): Topic, domain tag, publication year bounds.

### 6.2 Expected Output
* `verdict` (enum): `SUPPORTED` | `REFUTED` | `PARTIALLY_SUPPORTED` | `NEUTRAL_INSUFFICIENT_EVIDENCE`
* `confidence_score` (float: 0.0 to 1.0)
* `risk_level` (enum): `LOW` | `MEDIUM` | `HIGH` | `VERY_HIGH`
* `evidence_attribution` (list of dicts):
  * `doc_id`, `section`, `sentence_text`, `relation` (`SUPPORT`/`REFUTE`/`NEUTRAL`), `score`
* `verification_trace` (dict): Details on tier executed (Level 1/2/3), agent debate transcript, historical failure matches found.
* `failure_signature` (optional dict, if failed/hallucinated): Failure type code, scope delta, recorded ledger ID.

---

## 7. Main System Modules

```
src/
├── data/           # Dataset loaders, schema parsers, and preprocessors (SciFact, CoVERt, SciClaimHunt)
├── retrieval/      # Sparse (BM25) & Dense (FAISS/SentenceTransformers) hybrid retriever, document indexer
├── models/         # Claim decomposer, NLI verifier, Multi-Agent debate (Prosecutor, Skeptic, Leader)
│   ├── ledger.py   # Persistent Hallucination Ledger & Failure Signature generator
│   ├── agents.py   # Multi-agent debate orchestration
│   └── verifier.py # Dynamic adaptive verification router
├── evaluation/     # Metrics: Accuracy, Precision, Recall, F1, AUROC, Calibration Error, RHR
└── utils/          # Configuration, logging, vector operations
```

---

## 8. Evaluation Metrics & Benchmark Requirements

1. **Claim Verification Metrics**:
   * Accuracy, Precision, Recall, Macro-F1 across `SUPPORTED`, `REFUTED`, and `NEUTRAL` classes.
2. **Evidence Retrieval Metrics**:
   * Recall@K, Mean Reciprocal Rank (MRR), Precision@K for evidence sentence selection.
3. **Hallucination Detection Metrics**:
   * AUROC, Precision-Recall AUC, Expected Calibration Error (ECE).
4. **Closed-Loop Evolution Metrics**:
   * **Repeated Hallucination Rate (RHR)**: Percentage of structural failure patterns repeated across evaluation rounds $R_1 \dots R_n$.
   * **RHR Reduction (%)**: $\frac{\text{RHR}_{\text{stateless}} - \text{RHR}_{\text{citeguard}}}{\text{RHR}_{\text{stateless}}} \times 100\%$

---

## 9. Implementation Status

- **Phase 1 (Setup)**: COMPLETE
- **Phase 2 (Specification)**: COMPLETE
- **Phase 3 (Data Pipeline)**: COMPLETE
- **Phase 4 (Hybrid Retrieval)**: COMPLETE
- **Phase 5 (Hallucination Ledger)**: COMPLETE
- **Phase 6 (Multi-Agent Verification)**: COMPLETE
- **Phase 7 (Evaluation & RHR Benchmarking)**: COMPLETE
- **Phase 8 (System Integration & Verification)**: COMPLETE (84/84 tests passing, CLI entry point verified, full integration report in `docs/PHASE8_INTEGRATION_REPORT.md`).

