"""Post-hoc calibration, fit on the dedicated cal split and applied to test predictions.

Platt scaling (Platt 1999): logistic regression on the model's logit, two parameters, monotone,
so ranking metrics (ROC AUC) are unchanged. Isotonic regression (Zadrozny & Elkan 2002): a free
monotone step function; more flexible, needs more data, can introduce ties.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds with clipping so 0/1 probabilities stay finite."""
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(logit(p)[:, None])[:, 1]


class IsotonicCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}
