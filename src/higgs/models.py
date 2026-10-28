"""Model families. Each `fit_*` takes standardized arrays and returns a fitted Model.

Logistic regression (interpretable linear baseline) and gradient-boosted trees (XGBoost).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class Model(Protocol):
    info: dict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Signal probability for each row, shape (n,)."""


class SklearnLike:
    """Adapter so sklearn/xgboost estimators return a 1-D signal probability."""

    def __init__(self, est, info: dict):
        self.est = est
        self.info = info

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1]


def fit_logreg(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> SklearnLike:
    """L2 logistic regression; the interpretable linear baseline."""
    est = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=seed)
    est.fit(X_tr, y_tr)
    return SklearnLike(est, {"n_iter": int(est.n_iter_[0])})


def make_xgb(params: dict, seed: int) -> XGBClassifier:
    """XGBClassifier with the project's fixed choices (hist trees, log-loss early stopping)."""
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params.get("min_child_weight", 1),
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params.get("reg_lambda", 1.0),
        early_stopping_rounds=params["early_stopping_rounds"],
        n_jobs=params["n_jobs"],
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
    )


def fit_xgboost(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> SklearnLike:
    """Histogram gradient-boosted trees, early-stopped on validation log loss."""
    est = make_xgb(params, seed)
    est.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return SklearnLike(est, {"best_iteration": int(est.best_iteration)})


FITTERS = {"logreg": fit_logreg, "xgboost": fit_xgboost}
