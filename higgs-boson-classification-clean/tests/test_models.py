"""
Unit tests for higgs.models and higgs.evaluation modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from higgs.data import TARGET_COLUMN, load_higgs_sample, split_dataset
from higgs.evaluation import calculate_metrics, compile_comparison_table
from higgs.models import (
    LogisticRegressionModel,
    MultilayerPerceptronModel,
    XGBoostModel,
)


@pytest.fixture
def small_split_data():
    """Provides small train/val/test splits for fast unit testing."""
    df = load_higgs_sample(n_rows=400)
    return split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42)


def test_logistic_regression_fit_predict(small_split_data):
    """Verify LogisticRegressionModel fits, predicts probabilities, and extracts coefficients."""
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = small_split_data

    model = LogisticRegressionModel(max_iter=100, random_state=42)
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)
    preds = model.predict(X_te, threshold=0.5)

    assert len(probs) == len(X_te)
    assert len(preds) == len(X_te)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert set(preds).issubset({0.0, 1.0})

    coefs = model.get_coefficients()
    assert len(coefs) == X_tr.shape[1]
    assert "Feature" in coefs.columns
    assert "Coefficient" in coefs.columns


def test_xgboost_fit_predict(small_split_data):
    """Verify XGBoostModel fits with early stopping, predicts probabilities, and extracts importances."""
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = small_split_data

    model = XGBoostModel(n_estimators=10, max_depth=3, random_state=42, n_jobs=1)
    model.fit(X_tr, y_tr, X_val, y_val)

    probs = model.predict_proba(X_te)
    preds = model.predict(X_te, threshold=0.5)

    assert len(probs) == len(X_te)
    assert len(preds) == len(X_te)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    importances = model.get_feature_importances()
    assert len(importances) == X_tr.shape[1]
    assert "Importance" in importances.columns


def test_mlp_fit_predict(small_split_data):
    """Verify MultilayerPerceptronModel fits, tracks validation AUC, and predicts probabilities."""
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = small_split_data

    model = MultilayerPerceptronModel(
        hidden_dims=[32, 16],
        epochs=3,
        batch_size=64,
        random_state=42,
        device="cpu",
    )
    model.fit(X_tr, y_tr, X_val, y_val)

    probs = model.predict_proba(X_te)
    preds = model.predict(X_te, threshold=0.5)

    assert len(probs) == len(X_te)
    assert len(preds) == len(X_te)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert len(model.history) == 3


def test_calculate_metrics():
    """Verify evaluation metrics computation."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.7, 0.9])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 4


def test_compile_comparison_table():
    """Verify comparison table compilation."""
    records = [
        {"model_name": "Logistic Regression", "feature_set": "all", "feature_count": 28, "roc_auc": 0.67, "accuracy": 0.62, "precision": 0.61, "recall": 0.63, "f1": 0.62},
        {"model_name": "XGBoost", "feature_set": "all", "feature_count": 28, "roc_auc": 0.78, "accuracy": 0.71, "precision": 0.70, "recall": 0.73, "f1": 0.71},
    ]
    df_table = compile_comparison_table(records)
    assert len(df_table) == 2
    assert df_table.iloc[0]["Model"] == "XGBoost"  # Sorted by ROC AUC descending
