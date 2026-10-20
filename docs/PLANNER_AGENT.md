# CiteGuard PlannerAgent Architecture & Implementation Guide

## 1. Executive Summary & Purpose

In early scientific verification systems, candidate claims were generated prior to literature retrieval (*question-first claim generation*). This created an architectural vulnerability where language models could synthesize plausible-sounding but ungrounded hypotheses, exposing the downstream verification engine to confirmation bias.

The **PlannerAgent** resolves this architectural issue by serving as the first stage in CiteGuard's target scientific verification pipeline:

$$\text{User Research Question} \longrightarrow \mathbf{PlannerAgent} \longrightarrow \text{Research Plan} \longrightarrow \text{Paper Retrieval} \longrightarrow \text{Claim Extraction} \longrightarrow \dots$$

The PlannerAgent converts unstructured scientific research questions into a structured, evidence-neutral **Research Plan** without inventing scientific evidence, fabricating citations, or predetermining verdicts.

---

## 2. Research Architecture Target

```text
USER RESEARCH QUESTION
        ↓
  PLANNER AGENT
  ├── Objective Formulation
  ├── Question Type Classification & Enrichment
  ├── Subquestion Decomposition (4–6 targeted inquiries)
  ├── Targeted Retrieval Query Generation (OpenAlex / BM25 / FAISS)
  └── Evidence Requirements & Constraint Identification
        ↓
  PAPER RETRIEVAL (OpenAlex API + Local SciFact/SciClaimHunt Corpus)
        ↓
  CLAIM EXTRACTION (Retrieval-first evidence grounding)
        ↓
  CLAIM + EXACT EVIDENCE PASSAGES
        ↓
  CLAIM-EVIDENCE ALIGNMENT
        ↓
  ADVERSARIAL SKEPTIC AGENT (F1–F8 Challenge)
        ↓
  VERIFICATION AGENT / MULTI-AGENT CONSENSUS
        ↓
  CANONICAL F1–F8 FAILURE CLASSIFIER
        ↓
  FAILURE-PATTERN MEMORY & HISTORICAL MATCHING
        ↓
  ADAPTIVE VERIFICATION & CLAIM RE-CHECK
        ↓
  ACCEPT / REVISE / REJECT FINAL ANSWER
```

---

## 3. Structured Output: `PlannerResult`

The PlannerAgent outputs an authoritative `PlannerResult` dataclass that is 100% JSON-serializable:

```python
@dataclass
class PlannerResult:
    original_question: str
    research_objective: str
    question_type: str
    research_subquestions: List[str]
    retrieval_queries: List[str]
    evidence_requirements: List[str]
    expected_claim_types: List[str]
    population_constraints: List[str]
    domain_constraints: List[str]
    temporal_constraints: Dict[str, Any]
    comparison_targets: Optional[List[str]]
    plan_confidence: float = 0.90
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 4. Question Classification & Semantic Enrichment

The PlannerAgent preserves and enriches CiteGuard's canonical question taxonomy from `src/claim_generation/classifier.py`:

| Canonical Question Type | Description | Key Planning Strategy |
| :--- | :--- | :--- |
| **`SCIENTIFIC_HYPOTHESIS`** | Cause-and-effect or empirical relationships between variables $X$ and $Y$. | Formulates causal vs correlational requirements, population scope, and effect direction checks. |
| **`COMPARATIVE_SCIENTIFIC`** | Comparisons across architectures, models, interventions, or configurations. | Identifies baseline vs candidate targets, standardized benchmarks, and trade-off metrics. |
| **`BIBLIOGRAPHIC_SEARCH`** | Requests for top-cited papers, publication year filters, or exact DOIs. | Mandates CrossRef/OpenAlex registry verification, citation ranking metrics, and exact publication date matching. |
| **`LITERATURE_REVIEW`** | Broad topical survey or overview queries. | Targets foundational surveys, meta-analyses, and methodological consensus. |
| **`METADATA_REQUEST`** | Specific metadata lookups (DOI, author attribution, journal venues). | Focuses retrieval queries on bibliographic registry resolution. |

---

## 5. Decomposed Research Subquestions

The PlannerAgent systematically generates 4–6 meaningful, domain-specific subquestions that guide literature retrieval and evidence extraction:

### Example: Causal / Hypothesis Question
**Input**: *"Does regular exercise reduce cardiovascular disease risk?"*

1. *What specific outcomes or biomarkers representing cardiovascular disease risk have been empirically studied?*
2. *What study populations, experimental models, or test systems were evaluated for regular exercise?*
3. *What dosage, intensity, model parameterization, or duration of regular exercise were tested?*
4. *Is the published evidence primarily observational, randomized controlled, or mechanistic in vitro?*
5. *What effect direction (positive, inverse, null) is consistently reported between regular exercise and cardiovascular disease risk?*
6. *Are there documented contradictory, null, or statistically disputed findings in peer-reviewed literature?*

### Example: Comparative Question
**Input**: *"How does model size affect the accuracy and computational cost of large language models?"*

1. *What empirical outcomes or benchmark tasks have evaluated smaller parameter configurations against scaled parameter configurations?*
2. *Under what experimental or operational conditions was the comparison conducted?*
3. *What quantitative metrics (e.g. accuracy, latency, efficiency, throughput) distinguish the configurations?*
4. *What trade-offs between performance and computational or biological cost are reported?*
5. *Are there specific scenarios or configurations where the comparative advantage reverses?*
6. *Are there contradictory or statistically insignificant comparative findings in the literature?*

---

## 6. Multi-Target Retrieval Query Generation

To maximize retrieval recall and precision across OpenAlex, BM25, and FAISS vector indices, the planner generates targeted query variations stripped of conversational filler and boolean errors:

- **Core Query**: Punctuation-free primary question.
- **Topic Query**: Cleaned noun-phrase topic.
- **Variable Combinations**: E.g. `regular exercise cardiovascular disease risk`, `exercise cardiovascular mortality cohort`.
- **Methodological Keywords**: E.g. `systematic review meta-analysis`, `clinical trial cohort study`, `empirical evaluation benchmark`.

---

## 7. Evidence Requirements (Not Fabricated Evidence)

The PlannerAgent specifies what kind of evidence is required to answer the question, adhering to the principle: **Requirements, NOT evidence**:

- **Study Design**: Explicit empirical experimental design, RCT, or benchmark corpus.
- **Literature Grounding**: Peer-reviewed publication with verified index registry entry.
- **Bibliographic Provenance**: Exact registered DOI and verified author attribution (for bibliographic inquiries).
- **Causal Attribution**: Discrimination between observational association and mechanistic causation.
- **Contradiction Scrutiny**: Mandatory check against refuting or conflicting literature.

---

## 8. Pipeline Integration & Backward Compatibility

- **`ResearchPipeline` (`src/research_pipeline.py`)**:
  - `PlannerAgent.plan(research_question)` executes at Step 1.
  - The resulting `research_plan` dictionary is embedded in the return payload, attached to `ResearchQuestion.metadata["research_plan"]`, and populates `ResearchQuestion.sub_questions`.
  - Existing claim generation remains functional to maintain 100% backward compatibility with downstream verification tests until Phase 2 of claim extraction is introduced.
- **REST API (`POST /api/verify`)**:
  - Emits `"research_plan"` in the response payload.
- **UI Stepper & Results Dashboard**:
  - Stepper updated to 8 stages: Research Question $\rightarrow$ Planning $\rightarrow$ Paper Retrieval $\rightarrow$ Claim Extraction $\rightarrow$ Evidence Alignment $\rightarrow$ Skeptic Verification $\rightarrow$ Failure Memory $\rightarrow$ Final Answer.
  - Dashboard renders Section 0: Structured Research Plan with objective, subquestions, retrieval queries, and evidence requirements.

---

## 9. Limitations

1. **Retrieval-First Transition**: The current phase exposes `research_plan` to downstream modules; complete replacement of question-first claim generation with retrieval-first claim extraction is scheduled for the subsequent phase.
2. **Deterministic Heuristics**: While supporting LLM providers, offline planning relies on robust deterministic regex and semantic templates to guarantee reproducible evaluation without cloud API dependencies.
