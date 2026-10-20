# Phase 5: Hallucination Ledger & Structured Failure Signature Engine Report

**Project**: CiteGuard — Scientific Claim Verification Engine  
**Phase**: Phase 5 — Hallucination Ledger & Structured Failure Signature Engine  
**Status**: COMPLETED  
**Date**: September 5, 2026  

---

## Executive Summary

Phase 5 establishes a structured, deterministic, and persistent framework for identifying, categorizing, recording, and analyzing system failures and hallucinations within the CiteGuard scientific claim verification pipeline. 

Rather than treating all incorrect predictions as generic "hallucinations", Phase 5 enforces a controlled 10-category taxonomy (`FailureType`) and distinguishes between **Retrieval Failures** (where relevant evidence was not retrieved in top-K) and **Reasoning Failures** (where relevant evidence was retrieved but misclassified). All failure records are persisted in a deduplicated JSON Lines ledger (`results/hallucination_ledger/hallucination_ledger.jsonl`) with automated summary statistics and visualization table exports.

---

## 1. Files Created & Modified

### Primary Engine Modules
1. `src/evaluation/failure_taxonomy.py` — Defines `FailureType` Enum (10 categories) and taxonomy metadata descriptions.
2. `src/evaluation/failure_signature.py` — Implements dataclass `FailureSignature` with SHA256 signature hashing for duplicate prevention.
3. `src/evaluation/hallucination_ledger.py` — Implements `HallucinationLedger` persistent storage with query filtering, deduplication, JSONL serialization, and CSV export.
4. `src/evaluation/failure_analyzer.py` — Implements deterministic rule engine (`FailureAnalyzer`) with explicit helper functions `classify_retrieval_failure()` and `classify_reasoning_failure()`.
5. `src/evaluation/failure_statistics.py` — Computes metrics (total cases, accuracy, failure rates by dataset/label/retrieval mode) and generates visualization tables.

### Execution Scripts & Integration
6. `scripts/analyze_failures.py` — Evaluates claim verification outputs against BM25, Dense, and Hybrid retrieval modes, populates the ledger, and exports summary statistics.

### Comprehensive Test Suite
7. `tests/evaluation/test_taxonomy.py` — Validates enum values and controlled vocabulary definitions.
8. `tests/evaluation/test_signature.py` — Validates dataclass validation, dictionary serialization, and signature hash generation.
9. `tests/evaluation/test_ledger.py` — Tests insertion, duplicate prevention, multi-field filtering, JSONL persistence, and CSV export.
10. `tests/evaluation/test_analyzer.py` — Verifies classification rules for retrieval vs. reasoning failures, unknown fallbacks, and boundary conditions.
11. `tests/evaluation/test_statistics.py` — Verifies total counts, accuracy calculations, group-by failure rates, and visualization table formatting.

### Documentation
12. `docs/FAILURE_TAXONOMY.md` — Detailed reference guide for failure categories and diagnostic rules.
13. `docs/HALLUCINATION_LEDGER.md` — Technical reference for schema, ledger operations, storage formats, and decision trees.
14. `docs/PHASE5_FAILURE_ANALYSIS_REPORT.md` — Phase 5 completion report.
15. `README.md` — Updated with Phase 5 components, architecture diagram, and usage instructions.

---

## 2. Failure Taxonomy Summary

CiteGuard defines 10 controlled categories (`FailureType`):

| Failure Category | Code | Description |
| :--- | :--- | :--- |
| **No Evidence** | `NO_EVIDENCE` | No evidence retrieved above minimum threshold. |
| **Wrong Evidence** | `WRONG_EVIDENCE` | Retrieved evidence does not support claim analysis. |
| **Insufficient Evidence** | `INSUFFICIENT_EVIDENCE` | Evidence is inadequate to verify/refute claim. |
| **Contradictory Evidence** | `CONTRADICTORY_EVIDENCE` | Evidence directly contradicts system output. |
| **Partial Evidence** | `PARTIAL_EVIDENCE` | Evidence covers only a subset of claim scope. |
| **Retrieval Failure** | `RETRIEVAL_FAILURE` | Ground-truth evidence exists but omitted from top-K results. |
| **Reasoning Failure** | `REASONING_FAILURE` | Ground-truth evidence retrieved in top-K, but final label misclassified. |
| **Label Failure** | `LABEL_FAILURE` | System reasoning valid, but final label misassigned. |
| **Source Quality Failure** | `SOURCE_QUALITY_FAILURE` | Retrieved source is unverified or noisy. |
| **Unknown Failure** | `UNKNOWN_FAILURE` | Default fallback when cause cannot be deterministically proved. |

---

## 3. Ledger Schema & Deduplication

The `FailureSignature` schema captures metadata required for error diagnosis and research reproducibility:

- `claim_id` & `claim_text`
- `dataset` & `split`
- `predicted_label` & `ground_truth_label`
- `retrieved_document_ids` & `retrieved_ranks`
- `retrieval_mode` (`bm25`, `dense`, `hybrid`)
- `evidence_available` & `evidence_retrieved`
- `failure_type` & `failure_subtype`
- `confidence` & `explanation`
- `timestamp`, `run_id`, `metadata`
- `signature_hash`: Calculated via `sha256(run_id_dataset_claimId_mode_failureType)`

---

## 4. Test Execution & Verification Results

All unit tests in `tests/` were executed using `pytest` against synthetic test fixtures:

- **Total Test Files**: 8 test modules (3 existing retrieval/preprocessing tests + 5 new evaluation tests)
- **Total Unit Tests**: All tests passed cleanly (100% pass rate).
- **Execution Time**: ~90s (including vector index loading during full suite).

### Key Logic Verified:
1. Controlled vocabulary enum immutability and complete dictionary mapping.
2. Signatures enforce missing field defaults without error.
3. Ledger prevents duplicate insertions when re-evaluating the same run.
4. Ledger filters accurately by `dataset`, `failure_type`, `retrieval_mode`, `run_id`, `predicted_label`, and `ground_truth_label`.
5. Analyzer correctly identifies `RETRIEVAL_FAILURE` when ground-truth doc ID is missing from top-K.
6. Analyzer correctly identifies `REASONING_FAILURE` when ground-truth doc ID is present in top-K.
7. Statistics module computes exact failure percentages and produces structured CSV visualization data tables.

---

## 5. Integration Status with Phase 4 Retrieval Outputs

- `scripts/analyze_failures.py` integrates directly with:
  - Processed claim datasets (`data/processed/scifact/claims_val.jsonl`, etc.)
  - Vector & BM25 indices (`models/indexes/bm25_index.pkl`, `models/indexes/faiss_index.bin`)
- Analyzes verification predictions across `bm25`, `dense`, and `hybrid` retrieval modes.
- Appends generated failure signatures into `results/hallucination_ledger/hallucination_ledger.jsonl`.
- Outputs summary breakdown tables into `results/hallucination_ledger/`:
  - `failure_type_distribution.csv`
  - `dataset_failure_distribution.csv`
  - `retrieval_mode_comparison.csv`
  - `label_failure_distribution.csv`

---

## 6. Known Limitations & Next Steps

1. **Ground-Truth Requirement**: Distinguishing `RETRIEVAL_FAILURE` from `REASONING_FAILURE` requires ground-truth document IDs. Datasets lacking explicit document-level ground truth default to `UNKNOWN_FAILURE`.
2. **Phase 6 Readiness**: The failure signatures and ledger generated in Phase 5 provide the requisite diagnostics for Phase 6 (Multi-Agent Debate Pipeline), which will target reasoning failures by cross-examining retrieved evidence.

---

**Conclusion**: Phase 5 is fully implemented, verified, and documented. Standing by for user approval prior to proceeding to Phase 6.
