# Comprehensive Dataset Registry & Completeness Audit

> [!IMPORTANT]
> **Data Integrity Policy**: 
> 1. Raw dataset files under `data/raw/` are preserved as read-only source files.
> 2. Seed/test datasets (used strictly for offline unit testing) are clearly separated from full experimental datasets.
> 3. Synthetic/seed records MUST NOT be used for final model evaluation or academic benchmarking.

---

## 1. Dataset Completeness Audit Summary

| Dataset Name | Official Expected Dataset | Current Raw Files Present | Record Count | Full Dataset Size | Complete Status | Access / Download Method |
|:---|:---|:---|:---|:---|:---|:---|
| **SciFact** | SciFact Scientific Claim Release v1.0 | `claims_train.jsonl`, `claims_dev.jsonl`, `claims_test.jsonl`, `corpus.jsonl` | **2,003** claim records (809 train, 300 dev, 300 test claims; 5,183 corpus abstracts) | ~1.4K claims / 5,183 abstracts (~8.5 MB) | **YES (COMPLETE)** | Direct automated download from Allen AI AWS S3 (`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`). |
| **CoVERT** | CoVERT Biomedical Tweet Corpus (Mohr et al., LREC 2022) | `covert.json` | **3** (Test Seed) | 300 annotated biomedical claims/tweets (~3 MB) | **NO (TEST SEED ONLY)** | Requires academic request & access agreement via University of Stuttgart IMS Portal. |
| **SciClaimHunt** | SciClaimHunt Benchmark (Kumar et al., 2025; arXiv:2502.10003) | `sciclaimhunt_train.jsonl`, `sciclaimhunt_dev.jsonl`, `sciclaimhunt_test.jsonl` | **3** (Test Seed) | ~110,000 instances / full paper texts (3.18 GB CSV on Hugging Face) | **NO (TEST SEED ONLY)** | Available via Hugging Face Hub `AnshulS/dataset_for_scicllaimhunt`. |

---

## 2. Detailed Dataset Profiles & Access Procedures

### 2.1 SciFact Dataset (Complete)
* **Status**: **COMPLETE (Full Official Release)**
* **Official Source**: Allen Institute for AI (AI2) – [https://github.com/allenai/scifact](https://github.com/allenai/scifact)
* **Direct Download URL**: `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`
* **Raw File Locations**: `data/raw/scifact/data/`
* **Ingested Processed File**: `data/processed/scifact/scifact_unified.jsonl` (2,003 records)
* **Description**: 1,409 expert-annotated scientific claims paired with 5,183 PubMed abstracts, labeled as `SUPPORT`, `CONTRADICT`, or `NEUTRAL` with exact sentence rationale indices.

---

### 2.2 CoVERT Dataset (Test Seed Present / Full Experimental Access Pending)
* **Status**: **TEST SEED ONLY (3 Records for Unit Testing)**
* **Official Source**: Institute for Natural Language Processing (IMS), University of Stuttgart – [Bioclaim Resource Portal](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/bioclaim/)
* **Paper Reference**: Mohr, Wührl, and Klinger (2022), *"CoVERT: A Corpus of Fact-checked Biomedical COVID-19 Tweets"*, LREC 2022.
* **Full Dataset Size**: 300 fact-checked biomedical tweets with crowdsourced evidence, named entity, and relation annotations.
* **Reason for Seed Separation**: The raw CoVERT corpus contains Twitter post IDs and annotated evidence text subject to Twitter API Terms of Service and University of Stuttgart academic licensing.
* **Exact Manual Download Procedure for Full Dataset**:
  1. Visit the official IMS Stuttgart Bioclaim resource page: `https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/bioclaim/`.
  2. Fill out the academic end-user request form for the CoVERT corpus.
  3. Upon receiving the official dataset archive (`covert.json`), place the file in:
     `data/raw/covert/covert.json`
  4. Run `python -m src.data.pipeline` to regenerate `data/processed/covert/covert_unified.jsonl` with all 300 full experimental records.

---

### 2.3 SciClaimHunt Dataset (Test Seed Present / Full Experimental Download Procedure)
* **Status**: **TEST SEED ONLY (3 Records for Unit Testing)**
* **Official Source**: Hugging Face Hub – `AnshulS/dataset_for_scicllaimhunt` / arXiv:2502.10003 (Kumar et al., 2025)
* **Hugging Face URL**: [https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt](https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt)
* **Direct File Download URL**: `https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/resolve/main/dataset.csv`
* **Full Dataset Size**: ~110,000 claims with full-text paper evidence spans (3.18 GB CSV file).
* **Reason for Seed Separation**: The full 3.18 GB dataset contains 110,000 instances. The 3-record files under `data/raw/sciclaimhunt/` were created strictly as unit-testing fixtures.
* **Procedure to Obtain Full Dataset**:
  * **Option A (Automated Python Command)**:
    ```powershell
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/resolve/main/dataset.csv', 'data/raw/sciclaimhunt/dataset.csv')"
    ```
  * **Option B (Manual Download)**:
    1. Navigate to `https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/blob/main/dataset.csv`.
    2. Download `dataset.csv` (3.18 GB) and place it in `data/raw/sciclaimhunt/dataset.csv`.
    3. Run `python -m src.data.pipeline` to parse and ingest the complete experimental dataset.

---

## 3. Strict Data Boundary Guidelines

1. **Unit Testing & Scaffolding**: Seed files (`data/raw/covert/covert.json`, `data/raw/sciclaimhunt/sciclaimhunt_*.jsonl`) are retained strictly for offline `pytest` execution.
2. **Experimental Evaluation**: All model benchmarking and RHR evaluation in later phases MUST use the complete datasets (SciFact 2,003 records, CoVERT 300 records, SciClaimHunt full dataset).
