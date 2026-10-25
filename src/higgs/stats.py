"""Uncertainty for ROC AUC on the shared test set: fast DeLong (DeLong et al. 1988; Sun & Xu 2014).

Every model is scored on the same 500k test events, so their AUCs are correlated. DeLong's method
gives each AUC's variance and the full covariance between AUCs, which yields confidence intervals
and *paired* z-tests that account for that correlation (a plain bootstrap would need ~1000 refits
of the metric per model).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def delong(preds: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs (k,) and their covariance (k, k) for k score vectors `preds` (k, n) on labels `y`."""
    preds = np.atleast_2d(preds).astype(np.float64)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = rankdata(pos, axis=1)
    ty = rankdata(neg, axis=1)
    tz = rankdata(np.hstack([pos, neg]), axis=1)
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n          # structural components for positives
    v10 = 1 - (tz[:, m:] - ty) / m      # ... and for negatives
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def auc_table(preds: dict[str, np.ndarray], y: np.ndarray, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model AUC with (1 - alpha) CI, and paired DeLong tests for every pair of models."""
    names = list(preds)
    aucs, cov = delong(np.vstack([preds[k] for k in names]), y)
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(np.diag(cov))
    ci = pd.DataFrame({"key": names, "auc": aucs, "auc_se": se, "auc_lo": aucs - z * se, "auc_hi": aucs + z * se})
    rows = []
    for i, j in combinations(range(len(names)), 2):
        var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
        diff = aucs[i] - aucs[j]
        zstat = diff / np.sqrt(var) if var > 0 else 0.0
        rows.append({"a": names[i], "b": names[j], "auc_a": aucs[i], "auc_b": aucs[j], "diff": diff,
                     "diff_lo": diff - z * np.sqrt(max(var, 0)), "diff_hi": diff + z * np.sqrt(max(var, 0)),
                     "z": zstat, "p_value": 2 * norm.sf(abs(zstat))})
    return ci, pd.DataFrame(rows)
