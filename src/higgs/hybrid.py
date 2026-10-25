"""Hybrid 2: stacking XGBoost and the MLP with a logistic meta-learner.

Rationale (measured, not assumed): at 250k events the MLP ranks events better (higher ROC AUC) while
XGBoost's probabilities are better calibrated, and the two make partly different errors (trees cut
axis-aligned regions; the MLP fits smooth interactions). A logistic regression on the two models'
logits, fit on the held-out cal split, learns how much to trust each and returns calibrated
probabilities by construction. Its two weights are directly interpretable.
The naive alternative (averaging probabilities) is reported alongside as the ablation.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from higgs.calibration import logit


class Stacker:
    def __init__(self, base: list[str]):
        self.base = base

    def fit(self, preds_cal: dict[str, np.ndarray], y_cal: np.ndarray) -> "Stacker":
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._design(preds_cal), y_cal)
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        return self.lr.predict_proba(self._design(preds))[:, 1]

    def weights(self) -> dict[str, float]:
        """Meta-learner coefficient on each base model's logit (plus the intercept)."""
        w = {m: float(c) for m, c in zip(self.base, self.lr.coef_[0])}
        w["intercept"] = float(self.lr.intercept_[0])
        return w

    def _design(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([logit(preds[m]) for m in self.base])


def average(preds: dict[str, np.ndarray], base: list[str]) -> np.ndarray:
    """Naive hybrid used as the ablation for stacking: mean of base probabilities."""
    return np.mean([preds[m] for m in base], axis=0)
