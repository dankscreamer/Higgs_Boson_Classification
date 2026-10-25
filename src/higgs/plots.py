"""Static report figures. Colour = model (fixed slots), facet = feature set."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# First three slots of the validated categorical palette (all-pairs CVD safe).
MODEL_COLORS = {"logreg": "#2a78d6", "xgboost": "#eb6834", "mlp": "#1baf7a"}
MODEL_LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP"}
FEATURE_MARKERS = {"low": "o", "geo": "D", "high": "s", "all": "^"}  # also the facet order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10,
})


def _facets(feature_sets: list[str]):
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(4.2 * len(feature_sets), 4.2),
                             sharex=True, sharey=True, squeeze=False)
    return fig, axes[0]


def plot_roc(preds: dict[str, np.ndarray], y: np.ndarray, out: Path) -> None:
    """ROC curves, one panel per feature set. `preds` keys are '<model>|<feature_set>'."""
    keys = [k.split("|") for k in preds]
    fsets = [f for f in FEATURE_MARKERS if any(fs == f for _, fs in keys)]
    fig, axes = _facets(fsets)
    for ax, fs in zip(axes, fsets):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
        for model, f in keys:
            if f != fs:
                continue
            fpr, tpr, _ = roc_curve(y, preds[f"{model}|{f}"])
            ax.plot(fpr, tpr, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.set_title(f"{fs} features", loc="left")
        ax.set_xlabel("Background efficiency (FPR)")
    axes[0].set_ylabel("Signal efficiency (TPR)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_reliability(preds: dict[str, np.ndarray], y: np.ndarray, out: Path, n_bins: int = 15) -> None:
    """Reliability diagrams (observed signal rate vs mean predicted probability) per feature set."""
    keys = [k.split("|") for k in preds]
    fsets = [f for f in FEATURE_MARKERS if any(fs == f for _, fs in keys)]
    fig, axes = _facets(fsets)
    for ax, fs in zip(axes, fsets):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Perfect calibration")
        for model, f in keys:
            if f != fs:
                continue
            frac, mean = calibration_curve(y, preds[f"{model}|{f}"], n_bins=n_bins)
            ax.plot(mean, frac, color=MODEL_COLORS[model], marker="o", ms=4, label=MODEL_LABELS[model])
        ax.set_title(f"{fs} features", loc="left")
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed signal fraction")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_auc_vs_time(results: pd.DataFrame, out: Path) -> None:
    """Accuracy/compute trade-off: test ROC AUC against training time (log scale)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for _, r in results.iterrows():
        ax.scatter(r.train_seconds, r.roc_auc, s=60, color=MODEL_COLORS[r.model],
                   marker=FEATURE_MARKERS[r.feature_set], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.annotate(r.feature_set, (r.train_seconds, r.roc_auc), xytext=(6, -11 if r.feature_set == "geo" else -3),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Test ROC AUC")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=MODEL_LABELS[m])
               for m, c in MODEL_COLORS.items() if m in set(results.model)]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
