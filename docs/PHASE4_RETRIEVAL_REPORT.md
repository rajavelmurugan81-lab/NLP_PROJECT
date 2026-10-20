# Phase 4: Hybrid Retrieval Engine & Document Vector Indexing Report

## 1. Task 0: Data Quality Safety Check & Audit Results

Before building the search corpus and retrieval indexes, a strict data quality safety audit was conducted to verify record handling:

* **Total Raw SciClaimHunt Records**: **110,002**
* **Records with Missing Claims**: **1,109** (Empty or whitespace-only claim strings)
* **Records Excluded from Claim Query Sets**: **1,109**
* **Exact Exclusion Rule**: `claim is None or claim.strip() == ""` or `evidence_text is None or evidence_text.strip() == ""`
* **Raw Dataset Integrity**: Raw dataset files under `data/raw/` remain **100% unmodified**.
* **Audit Distinction**: Downstream evaluation explicitly tracks `raw_records`, `valid_records`, and `excluded_records`.

---

## 2. Corpus Construction & Storage Estimates

* **Retained Corpus Size**: **10,186 scientific documents**
  * **SciFact Corpus**: 5,183 PubMed abstracts
  * **CoVERT Corpus**: 3 Biomedical Tweet contexts
  * **SciClaimHunt Benchmark Subset**: 5,000 research paper section rationales
* **Deduplication**: 0 semantically different documents merged.

### Task 12 Storage Safety & Estimate
* **Estimated FAISS Index Size**: 14.92 MB
* **Estimated BM25 Index Size**: 29.53 MB
* **Actual Disk Usage**:
  * `models/indexes/bm25_index.pkl`: **29.53 MB**
  * `models/indexes/faiss_index.bin`: **14.92 MB**
  * `models/indexes/doc_id_map.json`: **4.81 MB**

---

## 3. Retrieval Subsystem Performance

| Subsystem | Algorithm / Model | Storage Size | Build / Indexing Time | Retrieval Speed (per query) |
|:---|:---|:---|:---|:---|
| **Sparse Retriever** | BM25Okapi (`rank-bm25`) | 29.53 MB | 1.95s | ~0.005s |
| **Dense Retriever** | `all-MiniLM-L6-v2` + FAISS IndexFlatIP | 14.92 MB | 303.15s | ~0.015s |
| **Hybrid Retriever** | Reciprocal Rank Fusion ($k=60$) | - | Instant | ~0.020s |

---

## 4. Test Results & Verification

Executed complete pytest test suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests/
```

**Results**: **31 passed in 89.98s**
* [tests/retrieval/test_corpus.py](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/tests/retrieval/test_corpus.py) – Corpus construction & serialization (Passed)
* [tests/retrieval/test_sparse.py](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/tests/retrieval/test_sparse.py) – BM25 tokenization, search, & persistence (Passed)
* [tests/retrieval/test_dense.py](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/tests/retrieval/test_dense.py) – FAISS dense index building, vector search, & disk persistence (Passed)
* [tests/retrieval/test_hybrid.py](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/tests/retrieval/test_hybrid.py) – RRF score computation & ablation mode selection (Passed)
* [tests/retrieval/test_metrics.py](file:///c:/Users/rajav/OneDrive/Desktop/nlp/nlp/tests/retrieval/test_metrics.py) – Recall@K, Precision@K, MRR, nDCG@K (Passed)

---

## 5. Ablation Study Modes

The system natively supports three retrieval modes for experimental comparison:
1. `mode="bm25"`: Keyword sparse search only.
2. `mode="dense"`: Semantic vector similarity search only.
3. `mode="hybrid"`: Sparse + Dense fusion using Reciprocal Rank Fusion (RRF).

---

## 6. System Limitations & Next Steps

* **Limitations**: High GPU acceleration is optional; CPU encoding of 10k documents takes ~5 minutes. Streaming index loading ensures low RAM footprint during inference.
* **Next Recommended Phase**: **Phase 5: Hallucination Ledger & Structured Failure Signature Engine** (`src/models/ledger.py`).
