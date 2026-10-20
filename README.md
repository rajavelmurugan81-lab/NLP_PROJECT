# CiteGuard: Failure-Aware Adaptive Verification & Closed-Loop Hallucination Memory for Scientific Claim Verification

## Project Overview
**CiteGuard** is an advanced Natural Language Processing (NLP) framework designed for evidence-based scientific claim verification and Retrieval-Augmented Generation (RAG) hallucination mitigation. Unlike existing stateless fact-checking systems that evaluate claims in isolation, CiteGuard establishes a closed-loop memory architecture using a **Hallucination Ledger** and **Failure Signatures**. It remembers past structural verification mistakes and dynamically escalates verification depth for high-risk or historically failure-prone claim patterns.

---

## Key Objectives
1. **Structured Failure Taxonomy**: Categorize verification and generation errors into 8 explicit failure types (Unsupported Claim, Citation Mismatch, Overgeneralization, Causal Leap, Outdated Evidence, Context Mismatch, Contradictory Evidence, Evidence Misinterpretation).
2. **Persistent Hallucination Ledger**: Maintain a queryable vector memory of past failure signatures and context vectors.
3. **Adaptive Verification Escalation**: Dynamically route claim-evidence pairs across 3 verification tiers based on calculated risk (Level 1: NLI -> Level 2: Multi-Agent Debate -> Level 3: Cross-Model & Scope Audit).
4. **Fine-Grained Evidence Attribution**: Map atomic claim components directly to explicit scientific source sentences.
5. **Closed-Loop Evolution**: Evaluate and reduce the **Repeated Hallucination Rate (RHR)** across sequential evaluation passes.

---

## Planned Architecture
```
[Scientific Claim & Context Input]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Atomic Claim Decomposition & Retrieval      │
│  (Sparse BM25 + Dense FAISS Vector Search)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Historical Failure Matching (Ledger Search) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Adaptive Verification Router (Levels 1-3)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Verdict & Sentence-Level Provenance         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Failure Signature Generation & Ledger Update│
└──────────────────────────────────────────────┘
```

---

## Dataset Requirements
* **SciFact**: Primary abstract-level scientific claim verification dataset (2,003 records).
* **CoVERt**: Biomedical claim dataset for context and population mismatch evaluation (300 records).
* **SciClaimHunt**: Full-text section-wise evidence retrieval and numerical claim verification benchmark (110,002 records, 3.18 GB).

---

## Implementation Phases
* **Phase 1**: Workspace Setup & Environment Scaffolding (Completed)
* **Phase 2**: Project Analysis & Specification (Completed)
* **Phase 3**: Dataset Ingestion & Preprocessing Pipeline (Completed)
* **Phase 4**: Hybrid Evidence Retriever & Vector Indexing Module (Completed)
* **Phase 5**: Hallucination Ledger & Failure Signature Engine (Completed)
* **Phase 6**: Adaptive Multi-Agent Verification Pipeline (Completed)
* **Phase 7**: Evaluation Framework & RHR Benchmarking (Completed)
* **Phase 8**: System Integration & Final Verification (Completed)
* **Phase 9**: Final Experimental Evaluation & Research Benchmarking (Completed)

---

## Quick Start & Usage Instructions

### 1. Environment Activation
```powershell
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 2. Main System Verification CLI (Phase 8 End-to-End Execution)
```powershell
# Run integrated scientific claim verification pipeline
python scripts/citeguard_cli.py --claim "MicroRNA-21 regulates apoptosis in cancer cells" --pretty
```

### 3. Master Experimental Evaluation & Plot Generation (Phase 9)
```powershell
# Run full experimental evaluation suite across all baselines & metrics
python scripts/run_phase9_experiments.py

# Generate publication-quality figures
python scripts/generate_plots.py
```

### 4. Run Retrieval Indexer & Search
```powershell
python scripts/build_indexes.py --max-sch-records 5000
python scripts/test_retrieval.py --query "CRISPR-Cas9 enables targeted DNA modification in human cell lines" --top-k 5
```

### 5. Run Complete Test Suite (84/84 Passing)
```powershell
python -m pytest tests/
```

### 6. Results Location
- Experimental Reports: [PHASE9_EXPERIMENTAL_EVALUATION.md](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/docs/PHASE9_EXPERIMENTAL_EVALUATION.md) & [PHASE9_RESULTS_REPORT.md](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/docs/PHASE9_RESULTS_REPORT.md)
- Machine-Readable Data: `results/retrieval/`, `results/verification/`, `results/ablation/`, `results/failure_analysis/`, `results/tables/`
- Publication Figures: `results/figures/` (`confusion_matrix.png`, `retrieval_comparison.png`, `component_ablation.png`, `per_class_f1.png`, `failure_distribution.png`, `rhr_reduction.png`)



