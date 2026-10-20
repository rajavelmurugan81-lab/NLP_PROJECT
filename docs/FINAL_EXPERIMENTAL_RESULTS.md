# CiteGuard: Final Experimental Results & Research Metrics

## 1. Overview

This document presents the complete empirical benchmark results, ablation study findings, failure signature statistics, and Repeated Hallucination Rate (RHR) evaluations for the **CiteGuard** scientific claim verification framework.

---

## 2. Experimental Benchmark Results Summary

### 2.1 Evidence Retrieval Performance (SciFact Benchmark)

| Retrieval System | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 Sparse** | 0.3200 | 0.5400 | 0.6400 | 0.2100 | 0.4210 |
| **FAISS Dense** | 0.3800 | 0.6200 | 0.7100 | 0.2400 | 0.4850 |
| **Hybrid RRF ($k=60$)** | **0.5100** | **0.7600** | **0.8400** | **0.2900** | **0.6120** |

### 2.2 System Verification Performance & Baselines

| System Baseline | Accuracy | Precision | Recall | Macro F1 | MRR | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 0.4000 | 0.1333 | 0.3333 | 0.1905 | 0.0000 | 0.0000 |
| **Sparse BM25 Baseline** | 0.4800 | 0.4500 | 0.4300 | 0.4200 | 0.4210 | 0.5400 |
| **Dense FAISS Baseline** | 0.5400 | 0.5100 | 0.5000 | 0.4900 | 0.4850 | 0.6200 |
| **Hybrid RRF Baseline** | 0.6200 | 0.6000 | 0.5700 | 0.5800 | 0.6120 | 0.7600 |
| **CiteGuard Full System** | **0.7400** | **0.7200** | **0.7000** | **0.7100** | **0.6120** | **0.7600** |

---

## 3. System Component Ablation Study

| Configuration | Component Added | Accuracy | Macro F1 | MRR | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. BM25 Only** | Sparse Retrieval | 0.4800 | 0.4200 | 0.4210 | 0.5400 |
| **B. Dense Only** | Dense FAISS | 0.5400 | 0.4900 | 0.4850 | 0.6200 |
| **C. Hybrid RRF** | BM25 + Dense RRF | 0.6200 | 0.5800 | 0.6120 | 0.7600 |
| **D. Full CiteGuard System** | Hybrid RRF + Multi-Agent Consensus | **0.7400** | **0.7100** | **0.6120** | **0.7600** |

---

## 4. Failure Signature Distribution (8 Taxonomy Categories)

| Failure Code | Category Description | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **F1** | Unsupported Claim | 4 | 26.7% |
| **F2** | Citation Mismatch | 2 | 13.3% |
| **F3** | Overgeneralization | 3 | 20.0% |
| **F4** | Causal Leap | 1 | 6.7% |
| **F5** | Outdated Evidence | 1 | 6.7% |
| **F6** | Context / Population Mismatch | 2 | 13.3% |
| **F7** | Contradictory Evidence | 1 | 6.7% |
| **F8** | Evidence Misinterpretation | 1 | 6.7% |

---

## 5. Closed-Loop Repeated Hallucination Rate (RHR)

| Framework | Repeat Failures | Total Pass-2 Failures | RHR (%) | RHR Reduction (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Stateless RAG Baseline** | 2 | 3 | 66.67% | 0.00% |
| **CiteGuard (Hallucination Ledger)** | 1 | 3 | **33.33%** | **50.00%** |

---

## 6. Generated Figures & Artifact Locations

- Machine-Readable Data: `results/final/metrics.json`, `results/tables/*.csv`
- Publication Figures: `results/figures/` (`confusion_matrix.png`, `retrieval_comparison.png`, `component_ablation.png`, `per_class_f1.png`, `failure_distribution.png`, `rhr_reduction.png`)
