# Hybrid Retrieval Engine Documentation

## 1. System Overview

CiteGuard implements a multi-stage **Hybrid Scientific Evidence Retrieval Engine** designed to fetch, rank, and attribute evidence for scientific claims across sparse (keyword-based) and dense (semantic-vector) representation spaces.

```
                    User Claim
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         Sparse Search        Dense Search
           (BM25)       (SentenceTransformers)
              │                   │
              ▼                   ▼
         Ranked Docs          Ranked Docs
              └─────────┬─────────┘
                        ▼
             Reciprocal Rank Fusion (RRF)
                        ▼
              Final Evidence Ranking
```

---

## 2. Component Architecture

### 2.1 Corpus Construction (`src/retrieval/corpus.py`)
* Assembles a unified evidence corpus from SciFact, CoVERT, and SciClaimHunt processed records.
* Assigns stable, unique document IDs (`doc_id`).
* Excludes invalid/empty records while maintaining exact audit counts (Task 0 Data Quality Safety Rule).

### 2.2 Sparse Retrieval (`src/retrieval/sparse.py`)
* **Algorithm**: `BM25Okapi` via `rank-bm25`.
* **Tokenization**: Custom scientific tokenizer (`tokenize_scientific_text`) that preserves chemical formulas (e.g. $H_2O$), statistical ranges ($p < 0.001$), hyphenated terms, and numerical entities.
* **Index Storage**: Pickled index at `models/indexes/bm25_index.pkl` (29.53 MB for 10,186 documents).

### 2.3 Dense Vector Retrieval (`src/retrieval/dense.py`)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Batch Processing**: Configurable batch encoding (default `batch_size=64`).
* **Vector Index**: `faiss.IndexFlatIP` (Inner Product / Cosine Similarity after vector normalization).
* **Index Storage**: Binary vector index at `models/indexes/faiss_index.bin` (14.92 MB for 10,186 vectors) and document mapping at `models/indexes/doc_id_map.json`.

### 2.4 Hybrid Reciprocal Rank Fusion (`src/retrieval/hybrid.py`)
* **RRF Formula**:
  $$RRF\_score(d) = \sum_{m \in M} \frac{1}{k + rank_m(d)}$$
  where $k$ is configurable (default $k=60$).
* **Ablation Modes**:
  1. `mode="bm25"`: BM25 sparse keyword search only.
  2. `mode="dense"`: Dense vector similarity search only.
  3. `mode="hybrid"`: Sparse + Dense fusion via RRF.

---

## 3. Configuration Parameters (`src/retrieval/config.py`)

| Setting | Default Value | Description |
|:---|:---|:---|
| `embedding_model_name` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace embedding model ID |
| `embedding_dimension` | `384` | Vector dimensionality |
| `batch_size` | `64` | Encoding batch size for dense indexer |
| `top_k_sparse` | `50` | Number of candidate docs retrieved by BM25 |
| `top_k_dense` | `50` | Number of candidate docs retrieved by FAISS |
| `top_k_final` | `10` | Final top-K evidence documents returned |
| `rrf_k` | `60` | Constant smoothing term in RRF formula |

---

## 4. Usage & Execution Instructions

### Build Retrieval Indexes
```powershell
python scripts/build_indexes.py --max-sch-records 5000
```

### Test Retrieval Engine
```powershell
python scripts/test_retrieval.py --query "CRISPR-Cas9 enables targeted DNA modification" --top-k 5
```
