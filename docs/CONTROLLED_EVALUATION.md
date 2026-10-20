# Controlled F1–F8 Failure Evaluation Framework

## 1. Overview
The Controlled Evaluation Suite (`src/evaluation/controlled_evaluation.py`) measures the diagnostic precision of CiteGuard across known, controlled failure archetypes spanning Canonical F1–F8:

- **F1: Unsupported Claim**: Evaluated on zero-evidence and subthreshold (< 0.70 score) configurations.
- **F2: Citation Mismatch**: Evaluated on cross-citation fabrications (e.g. GPT-3 claim supported by VGG ImageNet citation).
- **F3: Overgeneralization**: Evaluated on universal quantifiers ("all human populations worldwide") drawn from small/animal cohorts.
- **F4: Causal Leap**: Evaluated on causal assertions ("causes hypertension") supported by purely observational/correlational evidence.
- **F5: Outdated Evidence**: Evaluated on chronologically superseded evidence (> 25 years old or historical medical beliefs).
- **F6: Population / Context Mismatch**: Evaluated on in vitro cell-culture findings asserted as human clinical cures.
- **F7: Contradictory Evidence**: Evaluated on conflicting literature passages showing positive and null/negative outcomes.
- **F8: Evidence Misinterpretation**: Evaluated on inverted directional findings (claim asserts increase; evidence demonstrates impairment).

---

## 2. Experimental Results Summary

| Metric | Target Baseline | Achieved Value | Status |
| :--- | :--- | :--- | :--- |
| **Total Test Cases** | 9 | 9 | Covered F1–F8 |
| **Detection Rate** | $\\ge 90\\%$ | **100%** | PASSED |
| **Exact Category Match** | $\\ge 85\\%$ | **100%** | PASSED |
| **Zero False Support Rate** | 0.00 | **0.00** | PASSED |

---

## 3. Verification Command
```bash
python -m pytest tests/evaluation/test_controlled_evaluation.py -v
```
