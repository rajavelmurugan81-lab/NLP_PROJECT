# CiteGuard SkepticAgent Architecture & Implementation Guide

## 1. Executive Summary & Purpose

The **SkepticAgent** is an explicit adversarial verification agent in CiteGuard. While standard verification pipelines suffer from confirmation bias—seeking only supporting evidence and uncritically confirming plausibility—the SkepticAgent actively challenges scientific claims. It systematically examines evidence passages, citations, causal semantics, scope constraints, and domain contexts to find reasons why a claim should **NOT** be accepted.

---

## 2. Architectural Pipeline & Sequence

The SkepticAgent operates as the adversarial gatekeeper between preliminary multi-agent feature analysis and the authoritative consensus verification verdict:

```text
USER RESEARCH QUESTION
         ↓
  PAPER RETRIEVAL (OpenAlex / Hybrid BM25 + FAISS + RRF)
         ↓
  CLAIM EXTRACTION & PASSAGE ALIGNMENT
         ↓
  PRELIMINARY AGENTS
  ├── EvidenceAgent (Passage relevance, extraction)
  ├── ClaimAnalysisAgent (Syntactic complexity, entity breakdown)
  └── SourceQualityAgent (Peer-review provenance, DOI verification)
         ↓
  ADVERSARIAL SKEPTIC AGENT
  ├── Evaluates 8 Canonical Failure Modes (F1–F8)
  ├── Produces Structured SkepticResult
  └── Issues Objection Level: [NO_OBJECTION | POTENTIAL_FAILURE | CONFIRMED_FAILURE]
         ↓
  VERIFICATION AGENT / MULTI-AGENT CONSENSUS
  ├── Incorporates Skeptic Objections & Challenges
  └── Revises Decision & Grounds Confidence
         ↓
  CANONICAL F1–F8 FAILURE CLASSIFIER
  ├── Ingests Skeptic diagnostic signals
  └── Maps confirmed failures to authoritative Canonical Codes
         ↓
  FAILURE-PATTERN MEMORY & ADAPTIVE VERIFICATION
  └── Reuses failure signatures to reinforce future scientific checks
```

---

## 3. The 8 Canonical Failure Categories Evaluated by Skeptic

The SkepticAgent systematically evaluates candidate claims against all 8 canonical failure modes:

| Code | Failure Name | Skeptic Challenge & Evaluation Criteria | Severity |
| :--- | :--- | :--- | :--- |
| **F1** | **Unsupported Claim** | Flags claims asserting `SUPPORT` despite zero retrieved passages or where the highest evidence score falls below the strict 0.70 threshold. | `CRITICAL` / `HIGH` |
| **F2** | **Citation Mismatch** | Flags citations where DOI, author, or paper bibliographic metadata cannot be verified in the index registry. | `CRITICAL` |
| **F3** | **Overgeneralization** | Flags universal quantifiers (*"all"*, *"every"*, *"always"*, *"universally"*) asserted on narrow sample sizes ($< 3$ passages) or artificial high confidence on weak evidence. | `MEDIUM` |
| **F4** | **Causal Leap** | Flags claims asserting direct causality (*"causes"*, *"causally triggers"*, *"directly proves"*) when evidence only establishes observational or correlational association. | `HIGH` |
| **F5** | **Outdated Evidence** | Flags citations chronologically superseded or obsolete relative to the target research state of the art ($> 15$ years old or flagged superseded). | `MEDIUM` |
| **F6** | **Population / Context Mismatch** | Flags evidence retrieved from alien scientific domains or failing topic gating (e.g., smart grid literature retrieved for quantum federated learning). | `HIGH` |
| **F7** | **Contradictory Evidence** | Flags instances where contradictory or refuting evidence exists in the retrieved corpus but the claim is declared supported. Highest priority in objection hierarchy. | `CRITICAL` |
| **F8** | **Evidence Misinterpretation** | Flags distorted assertions, syntactic compound gerund malformations (*"altering increasing"*), or reversed directional relationships. | `HIGH` / `LOW` |

---

## 4. Structured Output: Three Objection Levels

The SkepticAgent emits a structured `SkepticResult` with three discrete objection levels:

```python
@dataclass
class SkepticResult:
    challenged: bool
    objection_level: str  # "NO_OBJECTION" | "POTENTIAL_FAILURE" | "CONFIRMED_FAILURE"
    failure_candidates: List[CanonicalFailureCode]
    primary_failure_code: Optional[CanonicalFailureCode]
    primary_failure_name: str
    confidence: float
    objections: List[str]
    supporting_evidence: List[Dict[str, Any]]
    contradicting_evidence: List[Dict[str, Any]]
    scope_analysis: Dict[str, Any]
    population_analysis: Dict[str, Any]
    causal_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    citation_analysis: Dict[str, Any]
    reasoning_analysis: Dict[str, Any]
    recommended_action: str
    metadata: Dict[str, Any]
```

### 1. `NO_OBJECTION`
- **Criteria**: Sufficient evidence ($\ge 1$ strong direct passage with score $\ge 0.70$), no contradictions, valid domain/topic gating, verified citations, and no ungrounded causal leaps.
- **Agent Decision**: `AgentDecision.SUPPORT`
- **Recommended Action**: `ACCEPT`

### 2. `POTENTIAL_FAILURE`
- **Criteria**: Subthreshold evidence relevance ($< 0.50$ floor) without an explicit `SUPPORT` declaration, minor scope ambiguity, or preliminary speculative assertions.
- **Agent Decision**: `AgentDecision.PARTIAL`
- **Recommended Action**: `MANDATE_STRONGER_VERIFICATION`

### 3. `CONFIRMED_FAILURE`
- **Criteria**: Hard violation detected across any canonical failure mode (F7 contradiction, F1 empty/subthreshold support, F2 citation mismatch, F4 causal leap, F6 off-domain evidence, F5 outdated evidence, F8 misinterpretation).
- **Agent Decision**:
  - `AgentDecision.REFUTE` (for F7 contradiction)
  - `AgentDecision.NEI` (for F1 unsupported, F2 citation mismatch, F6 context mismatch)
  - `AgentDecision.PARTIAL` (for F3 overgeneralization, F4 causal leap, F5 outdated evidence, F8 misinterpretation)
- **Recommended Action**: Specific canonical action from `CANONICAL_FAILURE_REGISTRY` (e.g. `RETRIEVE_EXPANDED_SCOPE`, `STRIP_UNGROUNDED_CAUSAL_CLAIMS`, `DEMOTE_CAUSAL_TO_CORRELATIONAL`).

---

## 5. Downstream Integration & Reversible Decision Flow

### VerificationAgent Revision
When `VerificationAgent.analyze(context)` runs, it inspects `context.get("skeptic_result")`. If a `CONFIRMED_FAILURE` is present:
- **Verdict Revision**: Updates its tentative decision (e.g., switches `SUPPORT` $\rightarrow$ `REFUTE` for F7, or `SUPPORT` $\rightarrow$ `NEI` for F1/F2/F6, or `PARTIAL` for F4).
- **Confidence Capping**: Caps verification confidence to $\le 0.60$ to reflect active skepticism.
- **Rationale Traceability**: Appends explicit `[Skeptic Revision: ...]` tags explaining the exact objection.

### MultiAgentOrchestrator Sequencing
The orchestrator strictly coordinates the 5-agent pipeline:
1. `EvidenceAgent` analyzes passage extraction and relevance scores.
2. `ClaimAnalysisAgent` inspects claim grammar and complexity.
3. `SourceQualityAgent` checks peer-review status and bibliographic attributes.
4. **`SkepticAgent`** runs adversarially with prior assessments in context, emitting `SkepticResult`.
5. `VerificationAgent` synthesizes all agent signals alongside the Skeptic's objections into a coherent verdict.

### Canonical FailureClassifier Feedback
The authoritative `FailureClassifier` ingests `skeptic_result`:
- Populates `diagnostic_signals["skeptic_objection_level"]`, `diagnostic_signals["skeptic_objections"]`, and `diagnostic_signals["skeptic_failure_candidates"]`.
- Ensures any confirmed failure identified by the SkepticAgent directly informs canonical F1–F8 classification and the persistent Hallucination Ledger.

---

## 6. Verification and Test Suite

The SkepticAgent test suite is located in `tests/verification/test_skeptic_agent.py` and provides 100% pass coverage for:
- Clean supported claim flow (`NO_OBJECTION`)
- All 8 canonical failure categories (F1 through F8)
- Potential vs. Confirmed failure levels
- VerificationAgent verdict revision
- FailureClassifier signal integration
- Orchestrator 5-agent sequential execution and metadata embedding
