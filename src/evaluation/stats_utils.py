"""Statistical significance testing and bootstrap confidence interval utilities."""

import random
import math
from typing import List, Dict, Any, Tuple
from src.evaluation.schema import Prediction
from src.evaluation.metrics import compute_classification_metrics


def bootstrap_confidence_interval(
    predictions: List[Prediction],
    metric_name: str = "macro_f1",
    n_bootstraps: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """Compute bootstrap confidence interval (lower, upper, mean) for a given metric."""
    if not predictions:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std_err": 0.0}

    rng = random.Random(seed)
    scores = []
    n = len(predictions)

    for _ in range(n_bootstraps):
        sample = [rng.choice(predictions) for _ in range(n)]
        metrics = compute_classification_metrics(sample)
        d = metrics.to_dict()
        val = float(d.get(metric_name, 0.0))
        scores.append(val)

    scores.sort()
    mean_val = sum(scores) / float(len(scores))
    
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * len(scores))
    upper_idx = int((1.0 - alpha) * len(scores)) - 1
    upper_idx = max(0, min(len(scores) - 1, upper_idx))

    lower_val = scores[lower_idx]
    upper_val = scores[upper_idx]

    variance = sum((x - mean_val) ** 2 for x in scores) / float(len(scores))
    std_err = math.sqrt(variance)

    return {
        "mean": round(mean_val, 4),
        "lower": round(lower_val, 4),
        "upper": round(upper_val, 4),
        "std_err": round(std_err, 4)
    }


def mcnemar_test(
    predictions_a: List[Prediction],
    predictions_b: List[Prediction]
) -> Dict[str, Any]:
    """Perform McNemar's test for paired classification predictions."""
    if len(predictions_a) != len(predictions_b):
        raise ValueError("Prediction lists must be of equal length for McNemar's test.")

    # Contingency table:
    # b_correct & a_correct (n00)
    # b_correct & a_wrong   (n01 - B right, A wrong)
    # b_wrong   & a_correct (n10 - A right, B wrong)
    # b_wrong   & a_wrong   (n11)
    b01 = 0  # A wrong, B right
    b10 = 0  # A right, B wrong

    for p_a, p_b in zip(predictions_a, predictions_b):
        gt = p_a.ground_truth_label.upper().strip()
        correct_a = (p_a.predicted_label.upper().strip() == gt)
        correct_b = (p_b.predicted_label.upper().strip() == gt)

        if correct_a and not correct_b:
            b10 += 1
        elif not correct_a and correct_b:
            b01 += 1

    total_discordant = b01 + b10
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b10": b10,
            "b01": b01,
            "note": "No discordant pairs found between systems."
        }

    # Continuity corrected McNemar statistic: (|b10 - b01| - 1)^2 / (b10 + b01)
    stat = ((abs(b10 - b01) - 1.0) ** 2) / float(total_discordant)
    
    # Approximate p-value using chi-square distribution with 1 degree of freedom
    # Normal approximation for chi-square(df=1): p ~ exp(-stat / 2)
    p_val = math.exp(-stat / 2.0)

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
        "b10_a_right_b_wrong": b10,
        "b01_a_wrong_b_right": b01
    }
