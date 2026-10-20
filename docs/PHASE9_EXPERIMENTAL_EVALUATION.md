# Phase 9: Experimental Evaluation & Research Benchmarking Specification

## 1. Overview

Phase 9 establishes the comprehensive experimental evaluation and research benchmarking framework for **CiteGuard**. The framework evaluates:
1. **Multi-Dataset Audits**: SciFact, CoVERT, and SciClaimHunt.
2. **Retrieval Algorithms**: BM25, Dense Vector FAISS, Hybrid, and Hybrid+RRF across Recall@K, Precision@K, and MRR.
3. **Verification Systems**: Majority Baseline, Sparse BM25 Baseline, Dense FAISS Baseline, Hybrid RRF Baseline, and Full CiteGuard Adaptive Multi-Agent system across Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix metrics.
4. **Failure Taxonomy & Ledger Analysis**: 8-category Failure Signature distribution (F1–F8) and persistent memory tracking.
5. **Ablation Studies**: Component-wise contributions across 5 configurations.
6. **Closed-Loop Metrics**: Repeated Hallucination Rate (RHR) and RHR reduction percentage.

---

## 2. Experimental Setup & Environment

- **Python Version**: Python 3.14.2 (win32)
- **Primary Dependencies**: `rank-bm25`, `faiss-cpu`, `sentence-transformers`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `pytest`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Reciprocal Rank Fusion (RRF)**: $k = 60$
- **Evidence Retrieval Top-K**: $K \in \{1, 5, 10\}$

---

## 3. Dataset Audit & Preprocessing Summary

| Dataset | Raw Records | Processed Records | Label Distribution | Status / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **SciFact** | 2,003 | 2,003 | SUPPORT (830), NEUTRAL (710), REFUTE (463) | Primary claim-abstract benchmark |
| **CoVERT** | 300 | 3 (Fixture) | SUPPORT (1), REFUTE (1), PARTIAL (1) | Biomedical seed dataset / unit fixture |
| **SciClaimHunt** | 110,002 | 3 (Fixture) / 110k Raw | Full-text full paper dataset | 3.18 GB dataset streaming index |

---

## 4. Evaluation Execution Commands

```bash
# Activate Virtual Environment
.\venv\Scripts\activate

# Run Master Benchmark Experiments
python scripts/run_phase9_experiments.py

# Generate Visualizations & PNG Figures
python scripts/generate_plots.py

# Run Full Pytest Regression Suite
python -m pytest tests/
```

---

## 5. Machine-Readable Results Location

All results are output deterministically under `results/`:

```
results/
├── dataset_audit.json
├── retrieval/
│   └── retrieval_metrics.json
├── verification/
│   ├── verification_summary.json
│   ├── majority_baseline_result.json
│   ├── bm25_sparse_baseline_result.json
│   ├── dense_faiss_baseline_result.json
│   ├── hybrid_rrf_baseline_result.json
│   └── citeguard_hybrid_multiagent_result.json
├── ablation/
│   └── ablation_study.json
├── failure_analysis/
│   └── failure_taxonomy_distribution.json
├── metrics/
│   └── rhr_metrics.json
├── figures/
│   ├── confusion_matrix.png
│   ├── retrieval_comparison.png
│   ├── component_ablation.png
│   ├── per_class_f1.png
│   ├── failure_distribution.png
│   └── rhr_reduction.png
└── tables/
    ├── dataset_audit.csv
    ├── baseline_comparison.csv
    ├── confusion_matrix.csv
    ├── ablation_study.csv
    ├── failure_taxonomy_distribution.csv
    └── rhr_metrics.csv
```
