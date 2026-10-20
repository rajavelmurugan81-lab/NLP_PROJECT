# CiteGuard: Rigorous Closed-Loop Failure-Pattern Learning Validation Experiment Design

## 1. Executive Summary & Objective

This document defines the formal experimental methodology, statistical hypotheses, dataset construction, system configurations, and validation criteria for rigorously evaluating CiteGuard's persistent closed-loop failure-pattern learning and prevention architecture.

The core scientific hypothesis is:
> **Hypothesis $H_1$**: Equipping the CiteGuard multi-agent scientific claim-verification pipeline with a persistent failure-pattern memory that mines runtime failure signatures into reusable prevention patterns will significantly reduce recurring verification failures on unseen held-out scientific claims without degrading accuracy on valid evidence or inducing false rejections.
>
> **Null Hypothesis $H_0$**: Persistent failure-pattern memory provides no statistically measurable improvement in verification accuracy or reduction in repeat failure rates over the memoryless baseline ($p \ge 0.05$ or $\Delta_{\text{repeat}} \le 0$).

---

## 2. Experimental Variables & Controls

### Independent Variables
- **System Configuration**:
  1. **Baseline**: Memoryless pipeline. Preliminary verification decisions and evidence rankings are executed without historical pattern matching or adaptive re-checks (`memory=None`).
  2. **CiteGuard (Full Closed-Loop)**: Includes persistent `FailureMemory` loaded with mined patterns, active semantic pattern matching, targeted prevention actions, and claim re-check verification gates.
  3. **Ablation (Memory without Re-check)**: Matches historical failure patterns but bypasses the adaptive verification/re-check gate, isolating whether pattern recognition alone or active intervention is necessary.

### Dependent Variables (Metrics)
1. **Verification Accuracy**: Proportion of held-out scientific claims whose final verdict matches the expert ground truth.
2. **Repeat Failure Count & Rate**: Number of failure-prone cases where the pipeline commits the known failure mode (e.g. accepting off-domain or correlational evidence as support).
3. **Relative Failure Reduction**:
   $$\Delta_{\text{reduction}} = \frac{\text{Failures}_{\text{baseline}} - \text{Failures}_{\text{CiteGuard}}}{\text{Failures}_{\text{baseline}}} \times 100\%$$
   *Engineering Success Threshold*: $\ge 20.0\%$.
4. **False Rejection Rate**: Rate at which legitimately supported claims are incorrectly downgraded or rejected.
   *Target*: Strictly $0.0\%$.
5. **False Positive Match Rate**: Rate at which benign/unrelated claims trigger false pattern matches.
   *Target*: Strictly $0.0\%$.
6. **Processing Latency (ms)**: End-to-end verification and pattern-matching overhead per claim.
7. **Benchmark Ledger Integrity**: Strict immutability of the historical benchmark ledger (`hallucination_ledger.jsonl`), maintaining exactly $N = 228$ records.

---

## 3. Dataset Specification (`closed_loop_validation_dataset.json` v1.0)

The evaluation dataset was constructed under strict provenance guidelines, pairing biomedical and computer science claims with controlled literature snippets:

### A. Learning Phase Cases ($N = 2$)
- **`learn_f6_001`**: Claim asserts human clinical efficacy for sulforaphane in glioblastoma, but retrieved evidence consists exclusively of *in vitro* cell culture assays (U87 human glioblastoma cell lines).
  - *Initial Failure*: Erroneously classified as SUPPORT.
  - *Adjudicated Ground Truth*: `NEI` (Population/Context Mismatch, F6).
- **`learn_f4_001`**: Claim asserts that dietary ultra-processed food intake *causes* major depressive disorder, but evidence is derived solely from an observational prospective cohort study.
  - *Initial Failure*: Erroneously classified as SUPPORT.
  - *Adjudicated Ground Truth*: `PARTIALLY_SUPPORTED` / `NEI` (Causal Leap, F4).

### B. Held-Out Evaluation Cases ($N = 13$)
Spanning all canonical failure modes, valid controls, and hard negatives:
1. **`test_f6_true_fail_001`** (F6 Test): Resveratrol extending human lifespan supported only by *in vitro* endothelial cell culture assays. (Ground Truth: `NEI`).
2. **`test_f6_hard_neg_001`** (Valid Clinical Control): Pembrolizumab improving overall survival in melanoma patients supported by Phase 3 clinical trial data. (Ground Truth: `SUPPORT`).
3. **`test_f4_true_fail_001`** (F4 Test): Coffee intake causing reduced Alzheimer's risk supported only by observational cross-sectional surveys. (Ground Truth: `PARTIALLY_SUPPORTED`).
4. **`test_f4_hard_neg_001`** (Valid Causal Control): Randomized double-blind controlled trial demonstrating PCSK9 inhibitors causally lower LDL cholesterol. (Ground Truth: `SUPPORT`).
5. **`test_f1_true_fail_001`** (F1 Test): Graphene water filtration claim with zero retrieved literature passages. (Ground Truth: `NEI`).
6. **`test_f1_valid_control`** (Valid Supported Control): Metformin first-line therapy for type 2 diabetes with strong clinical evidence. (Ground Truth: `SUPPORT`).
7. **`test_f7_true_fail_001`** (F7 Test): Hydroxychloroquine efficacy in COVID-19 contradicted by landmark RECOVERY trial. (Ground Truth: `REFUTE`).
8. **`test_f3_true_fail_001`** (F3 Test): Universal human cognitive enhancement extrapolated from 12 young healthy adults. (Ground Truth: `PARTIALLY_SUPPORTED`).
9. **`test_f5_true_fail_001`** (F5 Test): Historical superseded genomic guidelines from 1998. (Ground Truth: `NEI`).
10. **`test_f8_true_fail_001`** (F8 Test): Directional inversion claiming statins increase LDL cholesterol based on evidence showing statins reduce LDL. (Ground Truth: `REFUTE`).
11. **`test_f2_true_fail_001`** (F2 Test): Non-existent fabricated DOI bibliographic claim. (Ground Truth: `NEI`).
12. **`test_unrelated_physics_001`** (Orthogonal Control): Gravitational wave detection by LIGO/Virgo. (Ground Truth: `SUPPORT`).
13. **`test_unrelated_cs_001`** (Orthogonal Control): Transformer self-attention complexity scaling $O(n^2)$. (Ground Truth: `SUPPORT`).

---

## 4. Execution Protocol & Cold Process Restart

1. **Isolation Setup**: A temporary scratch directory is initialized containing a clean replica of the 10 core seed failure patterns. The historical benchmark ledger is checked and locked.
2. **Learning Execution**: The learning cases are ingested via `RuntimeAuditor.audit_claim()`.
3. **Mining Execution**: `PatternMiner.mine_patterns()` is invoked. The system must discover the novel failure category, instantiate a new pattern (`FP_PCM_011`), and persist it to JSON.
4. **Process Restart Validation**: A separate Python 3.14 subprocess is launched with a fresh Python runtime (no in-memory cache). The subprocess loads `FailureMemory` directly from disk and asserts that `FP_PCM_011` is immediately present and active.
5. **Held-Out Evaluation**: The 13 held-out test cases are run sequentially through Baseline, Full CiteGuard, and the Ablation pipeline.
6. **Integrity Audit**: The historical benchmark ledger is re-read to verify zero record mutations.
