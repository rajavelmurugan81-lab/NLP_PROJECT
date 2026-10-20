# Retrieval-First Claim Extraction & Provenance Tracking in CiteGuard

## 1. Executive Summary

CiteGuard enforces a foundational scientific invariant:
> **NO GROUNDED SCIENTIFIC CLAIM IS CREATED BEFORE RETRIEVAL.**

In conventional RAG systems, language models generate preliminary claims or answers prior to retrieving evidence, often anchoring subsequent retrieval around hallucinations or distorted premises. CiteGuard inverts this paradigm: claims are **extracted directly from retrieved, verified literature passages**, guaranteeing unambiguous provenance from origin to verdict.

---

## 2. Theoretical Architecture Flow

$$\\text{User Research Question} \\longrightarrow \\text{PlannerAgent} \\longrightarrow \\text{Hybrid Paper Retrieval} \\longrightarrow \\text{Exact Passage Extraction} \\longrightarrow \\mathbf{Retrieval-First\\ Claim\\ Extraction} \\longrightarrow \\text{Grounded Claims + Provenance} \\longrightarrow \\text{Claim-Evidence Alignment} \\longrightarrow \\text{SkepticAgent} \\longrightarrow \\text{Multi-Agent Verification} \\longrightarrow \\text{Canonical F1–F8 Classification} \\longrightarrow \\text{Failure Memory \\& Adaptive Recheck} \\longrightarrow \\text{Dual Ledger + Synthesized Provenance Report}$$

---

## 3. GroundedClaim Dataclass & Provenance Schema

Every extracted claim is modeled as a `GroundedClaim`, preserving full bibliographic and passage context:

```python
@dataclass
class GroundedClaim:
    claim_id: str                          # e.g., "CLM_P001_P01_001"
    claim_text: str                        # Extracted finding, stripped of meta-discourse
    source_paper_id: str                   # e.g., "W2963403869" or "doc_102"
    source_paper_title: str                # Full peer-reviewed title
    source_doi: Optional[str]              # Verified DOI link (e.g. "https://doi.org/10.48550/arXiv.1409.1556")
    source_passage_id: str                 # Exact passage ID (e.g. "P001_P01")
    exact_evidence_text: str               # Verbatim sentence-level passage
    retrieval_rank: int                    # Rank in hybrid RRF ranking
    evidence_score: float                  # RRF / cosine relevance score
    subquestion_alignment: Optional[str]   # Associated PlannerAgent subquestion
    claim_type: str                        # EMPIRICAL_OBSERVATION, CAUSAL_ASSERTION, etc.
    publication_year: Optional[int]        # Publication year for chronology checks
    extraction_confidence: float           # Confidence in factual sentence extraction
    provenance_status: str = "VERIFIED"    # Explicit provenance integrity status
```

### Duck-Type Compatibility
`GroundedClaim` exposes properties for `claim`, `text`, `paper_id`, `passage_id`, `doi`, `title`, and `exact_passage`. This ensures 100% backwards compatibility with components expecting `GeneratedClaim`.

---

## 4. Extraction Algorithm & Filtering Rules

1. **Topical Relevance Gate**: Candidate passages are scanned using token overlap against the research question, filtering out generic academic stop words (`dataset`, `results`, `shows`, `method`, etc.) to prevent false cross-domain alignment.
2. **Meta-Discourse Stripping**: Introductory academic framing phrases (`In this paper we show that`, `Our experimental findings demonstrate that`, `It was observed that`) are cleaned to yield concise, atomic factual assertions while preserving numerical values and directional semantics.
3. **Deterministic ID Generation**: Unique IDs link claims to their source paper and passage index (`CLM_<passage_id>_<idx>`).
4. **Fallback Safety**: If retrieval returns zero candidate literature passages (e.g. total network/index failure), the system flags `provenance_status="UNGROUNDED_FALLBACK"` and routes to conservative fallback generation.

---

## 5. Verification Baseline

- **Test Suite**: `tests/evaluation/test_retrieval_first_claim_extraction.py`
- **Result**: **20 passed, 0 failed**
- Verified properties:
  - 100% preservation of verbatim evidence text
  - Provenance status marked `VERIFIED`
  - Direct passage ID linkage (`P001_P01`)
  - Full propagation of DOI, authors, rank, and score
  - Resistance to cross-domain hallucinations (e.g., VGG queries returning zero false LLM claims)
