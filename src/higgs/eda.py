"""Data-quality checks and exploratory analysis, run on the training split only.

The physics checks encode expectations from the team's physics overview:
  * phi is an angle around the beam axis, and the collision is symmetric under rotation about it,
    so an absolute phi should carry ~no signal/background information (AUC ~ 0.5).
  * b-tagging is a per-jet classifier output, so those columns should take few discrete values.
  * signal is a heavy Higgs-like H -> W h -> W W b b; invariant masses of the true decay products
    should cluster near the parent mass for signal but spread out for the t-tbar background.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from higgs.data import HIGH_LEVEL  # noqa: E402
from higgs.plots import GRID, INK, MUTED  # noqa: E402  (also applies the shared rcParams)

SIGNAL, BACKGROUND = "#eb6834", "#2a78d6"          # categorical slots 2 and 1
LOW_COLOR, HIGH_COLOR = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#b8302f"])


def quality_report(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Counts of missing/non-finite values, duplicates, class balance and per-feature summaries."""
    df = pd.DataFrame(X, columns=names)
    per_feature = {}
    for name in names:
        col = df[name].to_numpy()
        n_unique = int(pd.Series(col).nunique())
        per_feature[name] = {
            "min": float(col.min()), "max": float(col.max()),
            "mean": float(col.mean()), "std": float(col.std()),
            "p01": float(np.quantile(col, 0.01)), "p99": float(np.quantile(col, 0.99)),
            "n_unique": n_unique, "frac_zero": float(np.mean(col == 0)),
            "discrete": n_unique <= 10,
        }
    return {
        "n_events": int(len(y)),
        "n_signal": int(y.sum()),
        "signal_fraction": float(y.mean()),
        "nan_values": int(np.isnan(X).sum()),
        "inf_values": int(np.isinf(X).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_features": [n for n, s in per_feature.items() if s["n_unique"] <= 1],
        "features": per_feature,
    }


def row_overlap(X_a: np.ndarray, X_b: np.ndarray) -> int:
    """Number of rows of X_b that also occur exactly in X_a (the UCI file contains duplicate events)."""
    hash_rows = lambda X: pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()  # noqa: E731
    return int(np.isin(hash_rows(X_b), hash_rows(X_a)).sum())


def single_feature_auc(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    """How well each feature separates the classes on its own (AUC folded so 0.5 = useless)."""
    rows = []
    for i, name in enumerate(names):
        auc = roc_auc_score(y, X[:, i])
        rows.append({"feature": name, "group": "high" if name in HIGH_LEVEL else "low",
                     "auc": max(auc, 1 - auc), "direction": "higher=signal" if auc >= 0.5 else "lower=signal"})
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


def mass_peaks(X: np.ndarray, y: np.ndarray, names: list[str], bins: int = 200) -> pd.DataFrame:
    """Histogram mode and IQR of each invariant mass, per class (scaled dataset units, not GeV)."""
    rows = []
    for name in HIGH_LEVEL:
        col = X[:, names.index(name)]
        hi = np.quantile(col, 0.995)
        for label, cls in (("signal", 1), ("background", 0)):
            v = col[y == cls]
            counts, edges = np.histogram(v, bins=bins, range=(0, hi))
            mode = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
            q25, q75 = np.quantile(v, [0.25, 0.75])
            rows.append({"feature": name, "class": label, "mode": mode, "median": float(np.median(v)),
                         "iqr": float(q75 - q25)})
    return pd.DataFrame(rows)


def plot_distributions(X: np.ndarray, y: np.ndarray, names: list[str], order: list[str], out: Path) -> None:
    """Signal vs background density for every feature, ordered by single-feature AUC."""
    ncols = 7
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 1.9 * nrows))
    for ax, name in zip(axes.flat, order):
        col = X[:, names.index(name)]
        vals = np.unique(col)
        if len(vals) <= 10:  # b-tag: one narrow bin centred on each discrete value
            bins = np.sort(np.concatenate([vals - 0.05, vals + 0.05]))
        else:
            bins = np.linspace(*np.quantile(col, [0.001, 0.995]), 60)
        for cls, color in ((0, BACKGROUND), (1, SIGNAL)):
            ax.hist(col[y == cls], bins=bins, density=True, histtype="step", color=color, linewidth=1.5)
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.grid(False)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=SIGNAL, label="Signal"),
               plt.Line2D([], [], color=BACKGROUND, label="Background")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Feature distributions by class (training split), most separating first",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_single_feature_auc(auc: pd.DataFrame, out: Path) -> None:
    """Horizontal bars: per-feature AUC, coloured by low-level vs expert feature group."""
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    d = auc.iloc[::-1]
    colors = [HIGH_COLOR if g == "high" else LOW_COLOR for g in d.group]
    ax.barh(d.feature, d.auc - 0.5, left=0.5, color=colors, height=0.7, edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_xlim(0.5, d.auc.max() + 0.02)
    ax.set_xlabel("ROC AUC of the feature alone (0.5 = no information)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HIGH_COLOR, label="Expert (high-level)"),
               plt.Rectangle((0, 0), 1, 1, color=LOW_COLOR, label="Raw (low-level)")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_correlation(X: np.ndarray, names: list[str], out: Path) -> pd.DataFrame:
    """Pearson correlation heatmap; returns the matrix."""
    corr = pd.DataFrame(X, columns=names).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    n_low = len(names) - len(HIGH_LEVEL)
    for pos in (n_low - 0.5,):
        ax.axhline(pos, color=INK, lw=0.8)
        ax.axvline(pos, color=INK, lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Pearson correlation")
    cb.outline.set_edgecolor(GRID)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return corr
