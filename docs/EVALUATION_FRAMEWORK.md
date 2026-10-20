# CiteGuard Evaluation Framework & Benchmark Architecture

## 1. Executive Overview

Phase 7 establishes a standardized, reproducible, and multi-dimensional evaluation engine for CiteGuard. The framework supports comparative benchmark evaluation across 5 system configurations:

1. `majority_baseline`: Fixed or majority class prediction heuristic.
2. `bm25_sparse_baseline`: Sparse BM25 retrieval + top-rank verdict heuristic.
3. `dense_faiss_baseline`: Dense FAISS vector search + top-rank verdict heuristic.
4. `hybrid_rrf_baseline`: Hybrid BM25 + FAISS + Reciprocal Rank Fusion (RRF).
5. `citeguard_hybrid_multiagent`: Hybrid RRF retrieval + adaptive multi-agent verification pipeline.

---

## 2. Evaluation Metrics

### 2.1 Information Retrieval Metrics
Evaluates retrieval subsystem quality against document-level ground truth:
- **Recall@K**: Proportion of ground-truth evidence passages retrieved in top-K.
- **Precision@K**: Proportion of top-K retrieved passages that are ground-truth evidence.
- **Hit@K**: Binary flag indicating whether at least 1 ground-truth passage appears in top-K.
- **MRR (Mean Reciprocal Rank)**: $1 / \text{rank}$ of the first retrieved ground-truth passage.
- **nDCG@K**: Discounted Cumulative Gain normalized by ideal DCG.

### 2.2 Classification & Verification Metrics
Evaluates end-to-end claim verification label accuracy:
- **Accuracy**: Total correct predictions over total evaluated cases.
- **Precision, Recall, Macro-F1, Micro-F1, Weighted-F1**: Calculated across multi-class labels (`SUPPORT`, `REFUTE`, `NEI`).
- **Confusion Matrix**: Full ground-truth vs. predicted label contingency matrix.

### 2.3 Expected Calibration Error (ECE)
Evaluates confidence score calibration by partitioning predictions into $M$ equal-width bins:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 2.4 Repeated Hallucination Rate (RHR) Methodology
CiteGuard measures closed-loop failure memory using **Repeated Hallucination Rate (RHR)**:
- **RHR Definition**: The proportion of failure signatures in evaluation pass $t$ whose SHA256 signature hash matches a historical failure signature logged in the `HallucinationLedger`:
$$\text{RHR} = \frac{\text{Count of Repeated Failure Hashes in Pass } t}{\text{Total Failures in Pass } t} \times 100\%$$
- **RHR Reduction (%)**:
$$\text{RHR Reduction (\%)} = \frac{\text{RHR}_{\text{stateless}} - \text{RHR}_{\text{citeguard}}}{\text{RHR}_{\text{stateless}}} \times 100\%$$

---

## 3. Experiment Directory Structure & Reproducibility

Every experiment run generates an isolated, immutable directory under `results/experiments/<run_id>/`:

```
results/experiments/<run_id>/
├── config.yaml          # Frozen experiment configuration
├── predictions.jsonl    # Line-by-line prediction objects with metadata
├── metrics.json         # Complete metric outputs (Retrieval, Classification, ECE, RHR)
├── errors.jsonl         # Phase 5 FailureSignature records
└── metadata.json        # System name, timestamp, sample limit, and completeness status
```

---

## 4. Running Benchmark Evaluations

### CLI Execution Examples:

```powershell
# Run evaluation across all 5 system baselines on SciFact (dev sample limit = 10)
python scripts/run_evaluation.py --dataset scifact --baseline all --limit 10

# Run evaluation on multi-agent system on CoVERT
python scripts/run_evaluation.py --dataset covert --baseline citeguard_hybrid_multiagent --limit 10

# Run full test suite (74 unit tests)
python -m pytest tests/
```
