"""Discrimination and calibration metrics computed on the held-out test set."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-bin ECE: weighted mean |observed signal rate - mean predicted probability|."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = np.bincount(bins, minlength=n_bins)
    pred_sum = np.bincount(bins, weights=p, minlength=n_bins)
    true_sum = np.bincount(bins, weights=y, minlength=n_bins)
    nonempty = total > 0
    gap = np.abs(true_sum[nonempty] - pred_sum[nonempty]) / total[nonempty]
    return float(np.sum(gap * total[nonempty]) / len(y))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """All headline metrics for predicted signal probabilities `p` against labels `y`."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, p >= 0.5),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }
