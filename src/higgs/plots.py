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

# Validated categorical palette, slots assigned in fixed order and never cycled. Slots 1-3 are
# all-pairs CVD safe; the hybrids (slots 4-6) only ever appear as lines (adjacent-pair safe).
MODEL_COLORS = {"logreg": "#2a78d6", "xgboost": "#eb6834", "mlp": "#1baf7a",
                "widedeep": "#eda100", "stack": "#e87ba4", "average": "#008300"}
MODEL_LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP",
                "widedeep": "Wide & deep (hybrid)", "stack": "Stacked XGB+MLP (hybrid)",
                "average": "Average XGB+MLP"}
# Also the facet order: raw -> geometry -> expert -> raw+expert -> geometry+expert.
FEATURE_MARKERS = {"low": "o", "geo": "D", "high": "s", "all": "^", "geo_high": "P"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10,
})


def _facets(feature_sets: list[str]):
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(3.8 * len(feature_sets), 4.0),
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
            if model == "average":
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
            if model == "average":
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
    """Accuracy/compute trade-off: test ROC AUC against training time (log scale).

    Colour = model, marker shape = feature set (two legends, so identity is never colour-alone).
    """
    results = results[results.model != "average"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, r in results.iterrows():
        ax.scatter(r.train_seconds, r.roc_auc, s=60, color=MODEL_COLORS[r.model],
                   marker=FEATURE_MARKERS[r.feature_set], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Test ROC AUC")
    models = [m for m in MODEL_COLORS if m in set(results.model)]
    fsets = [f for f in FEATURE_MARKERS if f in set(results.feature_set)]
    leg1 = ax.legend(handles=[plt.Line2D([], [], color=MODEL_COLORS[m], marker="o", ls="", label=MODEL_LABELS[m])
                              for m in models], frameon=False, loc="lower right", fontsize=8, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=[plt.Line2D([], [], color=MUTED, marker=FEATURE_MARKERS[f], ls="", label=f) for f in fsets],
              frameon=False, loc="lower center", fontsize=8, title="Features")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot_ablation_bars(summary: pd.DataFrame, out: Path) -> None:
    """Change in test ROC AUC vs the baseline MLP (mean ± sd over seeds), one panel per feature set."""
    fsets = list(dict.fromkeys(summary.feature_set))
    variants = list(dict.fromkeys(summary.variant))
    fig, axes = plt.subplots(1, len(fsets), figsize=(5.2 * len(fsets), 0.38 * len(variants) + 1.4),
                             sharey=True, squeeze=False)
    for ax, fs in zip(axes[0], fsets):
        g = summary[summary.feature_set == fs].set_index("variant").loc[variants[::-1]]
        ax.barh(g.index, g.delta_mean, xerr=g.delta_std.fillna(0), color=SLOTS[0], height=0.6,
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"{fs} features", loc="left")
        ax.set_xlabel("Δ test ROC AUC vs baseline")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_learning_curves(histories: dict, variants: list[str], fs: str, seed: int, out: Path) -> None:
    """Validation ROC AUC and training loss per epoch for selected ablation variants (one seed)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for color, v in zip(SLOTS, variants):
        h = histories.get(f"{v}|{fs}|{seed}")
        if not h:
            continue
        ep = [r["epoch"] for r in h]
        a1.plot(ep, [r["val_auc"] for r in h], color=color, label=v)
        a2.plot(ep, [r["train_loss"] for r in h], color=color, label=v)
    a1.set(xlabel="Epoch", ylabel="Validation ROC AUC", title=f"Validation AUC ({fs}, seed {seed})")
    a2.set(xlabel="Epoch", ylabel="Training BCE loss", title="Training loss")
    a1.title.set_ha("left"); a1.title.set_x(0)
    a2.title.set_ha("left"); a2.title.set_x(0)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, out: Path, title: str, top: int = 20) -> None:
    """Three importance measures side by side, features ordered by XGBoost mean |SHAP| (top `top`)."""
    imp = imp.head(top).iloc[::-1]
    cols = [("logreg_abs_coef", "logreg", "|coefficient| (standardised)"),
            ("xgb_mean_abs_shap", "xgboost", "mean |SHAP| (log-odds)"),
            ("mlp_perm_auc_drop", "mlp", "AUC drop when shuffled")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.3 * len(imp) + 1.6), sharey=True)
    for ax, (col, model, xlabel) in zip(axes, cols):
        ax.barh(imp.feature, imp[col], color=MODEL_COLORS[model], height=0.65)
        ax.set_title(MODEL_LABELS[model], loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"What each model relies on ({title})", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_shap_dependence(series: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    """SHAP value vs feature value for selected features; positive SHAP pushes towards signal."""
    fig, axes = plt.subplots(1, len(series), figsize=(4.6 * len(series), 3.8), squeeze=False)
    for ax, (name, (x, s)) in zip(axes[0], series.items()):
        lo, hi = np.quantile(x, [0.005, 0.99])
        keep = (x >= lo) & (x <= hi)
        ax.scatter(x[keep], s[keep], s=2, alpha=0.08, color=MODEL_COLORS["xgboost"], linewidths=0)
        edges = np.linspace(lo, hi, 40)
        idx = np.digitize(x[keep], edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        med = [np.median(s[keep][idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(edges))]
        ax.plot(centres, med, color=INK, lw=1.5, label="binned median")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(name, loc="left")
        ax.set_xlabel(f"{name} (dataset units)")
    axes[0][0].set_ylabel("SHAP contribution (log-odds)")
    axes[0][0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_scaling(scaling: pd.DataFrame, out: Path) -> None:
    """Test ROC AUC vs number of training events (log x), mean ± sd over seeds, one panel per feature set."""
    fsets = [f for f in FEATURE_MARKERS if f in set(scaling.feature_set)
             and scaling[scaling.feature_set == f].subset_size.nunique() > 1]
    fig, axes = _facets(fsets)
    for ax, fs in zip(axes, fsets):
        for model in ("logreg", "xgboost", "mlp"):
            g = scaling[(scaling.feature_set == fs) & (scaling.model == model)].sort_values("n_train")
            if g.empty:
                continue
            ax.errorbar(g.n_train, g.auc_mean, yerr=g.auc_sd.fillna(0), color=MODEL_COLORS[model],
                        marker=FEATURE_MARKERS[fs], ms=6, capsize=3, label=MODEL_LABELS[model])
        ax.set_xscale("log")
        ax.set_title(f"{fs} features", loc="left")
        ax.set_xlabel("Training events (log scale)")
    axes[0].set_ylabel("Test ROC AUC")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
