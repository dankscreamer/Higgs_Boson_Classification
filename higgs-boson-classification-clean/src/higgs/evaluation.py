"""
Evaluation metrics, comparison utilities, and diagnostic plotting for HIGGS classification.

At minimum calculates:
- ROC-AUC (threshold-independent discrimination)
- Accuracy at decision threshold
- Precision at decision threshold
- Recall at decision threshold
- F1 score
- Confusion matrix counts (TN, FP, FN, TP)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Computes comprehensive classification metrics from predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        True binary target labels (0.0 or 1.0).
    y_prob : array-like
        Predicted probabilities for the positive class (signal).
    threshold : float
        Decision boundary threshold for converting probabilities to hard binary classes.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'roc_auc': Area under the ROC curve
        - 'accuracy': Fraction of correct predictions at threshold
        - 'precision': True Positives / (True Positives + False Positives)
        - 'recall': True Positives / (True Positives + False Negatives)
        - 'f1': Harmonic mean of precision and recall
        - 'confusion_matrix': 2x2 ndarray [[TN, FP], [FN, TP]]
        - 'tn', 'fp', 'fn', 'tp': Integer counts
        - 'threshold': Evaluated decision threshold
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int32)

    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": threshold,
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluates a fitted model on holdout test data and computes all metrics."""
    y_prob = model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_prob, threshold=threshold)
    metrics["y_prob"] = y_prob
    metrics["y_true"] = np.asarray(y_test, dtype=np.int32)
    return metrics


def compile_comparison_table(results_records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Assembles a clean comparison DataFrame from experimental evaluation records.

    Parameters
    ----------
    results_records : list of dicts
        List of result dictionaries containing model name, feature set, and metrics.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    rows = []
    for r in results_records:
        rows.append(
            {
                "Model": r.get("model_name", "Unknown"),
                "Feature Set": r.get("feature_set", "Unknown"),
                "Features Count": r.get("feature_count", 0),
                "ROC AUC": r.get("roc_auc", 0.0),
                "Accuracy": r.get("accuracy", 0.0),
                "Precision": r.get("precision", 0.0),
                "Recall": r.get("recall", 0.0),
                "F1 Score": r.get("f1", 0.0),
            }
        )
    df_comp = pd.DataFrame(rows)
    return df_comp.sort_values(by="ROC AUC", ascending=False).reset_index(drop=True)


def plot_roc_curves(
    results_dict: dict[str, dict[str, Any]],
    title: str = "ROC Curves Across Models and Feature Groups",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Renders ROC curves comparing multiple models or feature representations.

    Parameters
    ----------
    results_dict : dict
        Mapping from label (e.g. 'XGBoost (all)') to evaluation dictionary
        containing 'y_true', 'y_prob', and 'roc_auc'.
    title : str
        Figure title.
    save_path : str | Path | None
        Optional filepath to export figure.

    Returns
    -------
    plt.Figure
        The rendered matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for label, res in results_dict.items():
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        auc_val = res["roc_auc"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2.0, label=f"{label} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=200, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Model",
    feature_set: str = "Features",
    class_names: Sequence[str] = ("Background", "Signal"),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Renders an annotated normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {model_name} [{feature_set}]", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=200, bbox_inches="tight")

    return fig
