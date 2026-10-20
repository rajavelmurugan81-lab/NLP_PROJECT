# SciClaimHunt Full Dataset Download & Verification Report

> [!IMPORTANT]
> **Experimental Data Integrity Notice**:
> The 3.18 GB official SciClaimHunt dataset has been fully acquired and verified. Seed fixtures for unit testing remain preserved under `data/raw/sciclaimhunt/fixtures/`.

---

## 1. Download & File Metadata

* **Official Dataset Source**: Hugging Face Hub – [`AnshulS/dataset_for_scicllaimhunt`](https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt)
* **Direct Source URL**: `https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/resolve/main/dataset.csv`
* **Downloaded Filename & Path**: `data/raw/sciclaimhunt/dataset.csv`
* **File Size**: **3,184,923,019 bytes** (~3.18 GB / 2.966 GiB)
* **Row Count**: **110,002 rows**
* **Format**: CSV

---

## 2. Dataset Schema & Structure

| Field Name | Data Type | Description & Purpose in CiteGuard |
|:---|:---|:---|
| `""` | int | Index identifier |
| `Type` | string | Claim classification (`positive` / `support` vs `negative` / `refute`) |
| `Claim` | string | Text of the scientific claim or numerical assertion |
| `Evidence` | string | Extracted evidence section / sentence rationale |
| `research_paper_full` | string | Full research paper text (used for full-text section retrieval & attribution) |

---

## 3. Streaming Statistical Analysis

The dataset was analyzed using memory-efficient streaming without loading the 3.18 GB file into RAM at once:

### 3.1 Quantitative Summary
* **Total Instances**: **110,002**
* **Unique Claims**: **76,563**
* **Duplicate Claims**: **33,439** *(Note: Multiple claims reference shared full research paper contexts)*

### 3.2 Missing Value Audit
* `missing_claim`: **1,109** (Empty claim fields, filtered out during ingestion)
* `missing_evidence`: **0**
* `missing_full_paper`: **0**
* `missing_type`: **0**

### 3.3 Text Length Profiling
| Attribute | Average Words | Average Characters |
|:---|:---|:---|
| **Scientific Claim** | **20.91 words** | **140.88 chars** |
| **Evidence Rationale** | **240.90 words** | **1,548.78 chars** |
| **Full Research Paper** | **4,498.08 words** | **27,126.70 chars** |

---

## 4. Memory Considerations & Implementation Architecture

To ensure high scalability and prevent RAM exhaustion during ingestion:
1. **Generator Streaming (`src/data/loader.py`)**: `SciClaimHuntLoader.stream(chunk_size=1000)` yields records iteratively using python `csv.DictReader` rather than loading all 110,002 full-paper records into RAM.
2. **Dynamic CSV Field Limit**: Configured `csv.field_size_limit(sys.maxsize)` to process full-text research paper fields (`research_paper_full`) exceeding 128 KB without buffer overflow.
3. **Fixture Separation**: 3-record unit-test files remain preserved under `data/raw/sciclaimhunt/fixtures/`. Offline `pytest` tests execute against fixtures in 0.10 seconds.

---

## 5. Preprocessing & Validation Status

* **Status**: **VERIFIED & PREPROCESSED**
* **Validation**: 0 malformed CSV syntax errors detected across all 110,002 rows.
* **Test Suite**: All 19 unit tests in `tests/` passed 100% cleanly.
