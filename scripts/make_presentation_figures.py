"""Slide-ready Phase 1 (ML-only) figures: logistic regression and XGBoost, 16:9, large type.

Reads the 250k-budget results (5 seeds) and seed 0's saved test predictions; writes
presentation/phase1_ml/figures/*.png. Run after scripts/aggregate.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.plots import INK, MODEL_COLORS, MUTED, plt  # noqa: E402  (shared palette + rcParams)
from higgs.metrics import expected_calibration_error  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "presentation" / "phase1_ml" / "figures"
BUDGET = 250000
FS_ORDER = ["low", "geo", "high", "all", "geo_high"]
FS_LABEL = {"low": "low · 21 raw", "geo": "geo · 40 geometric (ours)", "high": "high · 7 expert masses",
            "all": "all · raw + expert (paper)", "geo_high": "geo + high · 47"}
LABEL = {"logreg": "Logistic regression", "xgboost": "XGBoost (tuned)"}
# Calibration methods: validated categorical set (red/violet/green) + distinct marker shapes.
METHOD = {"ece": ("Raw", "#e34948", "o"), "ece_platt": ("Platt", "#6250d6", "s"),
          "ece_isotonic": ("Isotonic", "#008300", "D")}

plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 12})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(ROOT) / name}")


def title(ax, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", fontweight="bold", pad=28 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=12, va="bottom")


def auc_by_features(main: pd.DataFrame) -> None:
    """Dot plot: test ROC AUC per feature set, both models, mean ± sd over seeds."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(FS_ORDER))[::-1]
    for model, dy in (("logreg", -0.12), ("xgboost", 0.12)):
        g = main[main.model == model].set_index("feature_set").loc[FS_ORDER]
        ax.errorbar(g.roc_auc_mean, y + dy, xerr=g.roc_auc_std, fmt="o", ms=11, color=MODEL_COLORS[model],
                    mec="#fcfcfb", mew=2, capsize=3, label=LABEL[model], zorder=3)
        for yi, v in zip(y + dy, g.roc_auc_mean):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(12, 0), textcoords="offset points", va="center",
                        fontsize=12, color=INK)
    ax.set_yticks(y, [FS_LABEL[f] for f in FS_ORDER])
    ax.set_xlim(0.57, 0.875)
    ax.set_xlabel("Test ROC AUC  (500k fixed test events; mean ± sd over 5 seeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left")
    title(ax, "XGBoost beats logistic regression on every feature set",
          "Our geometric features (no masses) beat the physicists' 7 expert masses for XGBoost")
    save(fig, "1_auc_by_feature_set.png")


def significance_forest() -> None:
    """ΔAUC with 95% DeLong CI for the key ML comparisons (mean over 5 seeds)."""
    sig = pd.concat([pd.read_csv(f) for f in sorted(RES.glob(f"subset{BUDGET}_seed[0-4]/significance.csv"))])
    comps = [("xgboost|geo", "xgboost|low", "Geometric features vs raw  (XGBoost)"),
             ("xgboost|all", "xgboost|high", "Adding raw features to the masses  (XGBoost)"),
             ("xgboost|geo_high", "xgboost|all", "Geometry + masses vs paper's set  (XGBoost)"),
             ("xgboost|geo", "xgboost|high", "Geometric features vs expert masses  (XGBoost)"),
             ("logreg|geo", "logreg|low", "Geometric features vs raw  (LogReg)")]
    rows = []
    for a, b, label in comps:
        fwd = sig[(sig.a == a) & (sig.b == b)]
        rev = sig[(sig.a == b) & (sig.b == a)]
        d = pd.concat([fwd[["diff", "diff_lo", "diff_hi", "p_value"]],
                       pd.DataFrame({"diff": -rev["diff"], "diff_lo": -rev.diff_hi, "diff_hi": -rev.diff_lo,
                                     "p_value": rev.p_value})])
        rows.append({"label": label, "diff": d["diff"].mean(), "lo": d.diff_lo.mean(), "hi": d.diff_hi.mean(),
                     "n_sig": int(((d["diff"] > 0) & (d.p_value < 0.05)).sum()), "n": len(d),
                     "model": a.split("|")[0]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(df))[::-1]
    for yi, r in zip(y, df.itertuples()):
        ax.errorbar(r.diff, yi, xerr=[[r.diff - r.lo], [r.hi - r.diff]], fmt="o", ms=10, capsize=4,
                    color=MODEL_COLORS[r.model], mec="#fcfcfb", mew=2, zorder=3)
        ax.annotate(f"+{r.diff:.3f}   significant in {r.n_sig}/{r.n} seeds", (r.hi, yi), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=12, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, df.label)
    ax.set_xlim(-0.005, 0.11)
    ax.set_xlabel("Δ test ROC AUC  (A − B), with 95% DeLong confidence interval")
    ax.grid(axis="y", visible=False)
    title(ax, "Every improvement is statistically significant in all 5 seeds",
          "Paired DeLong test on the same 500k test events (accounts for correlated AUCs)")
    save(fig, "2_significance.png")


def tuning() -> None:
    """CV AUC of every XGBoost configuration tried, per feature set; chosen one highlighted."""
    cv = pd.read_csv(RES / "tuning" / "cv_results.csv")
    best = json.loads((RES / "tuning" / "best_params.json").read_text())["xgboost"]
    cv = cv[cv.model == "xgboost"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(FS_ORDER))[::-1]
    rng = np.random.default_rng(0)
    for yi, fs in zip(y, FS_ORDER):
        g = cv[cv.feature_set == fs]
        jitter = rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(g.cv_auc, yi + jitter, s=40, color=MUTED, alpha=0.55, linewidths=0, zorder=2,
                   label="Configurations tried" if yi == y[0] else None)
        top = g.cv_auc.max()
        ax.scatter([top], [yi], s=150, color=MODEL_COLORS["xgboost"], edgecolor="#fcfcfb", linewidth=2, zorder=3,
                   label="Selected" if yi == y[0] else None)
        p = best[fs]
        ax.annotate(f"depth {p['max_depth']}, η {p['learning_rate']}, mcw {p['min_child_weight']}  "
                    f"(+{top - g.cv_auc.min():.3f} over worst)", (top, yi), xytext=(14, 0),
                    textcoords="offset points", va="center", fontsize=11.5, color=INK)
    ax.set_yticks(y, [FS_LABEL[f] for f in FS_ORDER])
    ax.set_xlim(cv.cv_auc.min() - 0.01, cv.cv_auc.max() + 0.075)
    ax.set_xlabel("Mean 3-fold CV ROC AUC on the training split (test set never used)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper right")
    title(ax, "XGBoost tuning: 12-configuration random search per feature set",
          "Stratified 3-fold CV picks deep trees (depth 8–10), learning rate 0.05, min_child_weight 20")
    save(fig, "3_tuning.png")


def reliability() -> None:
    """Reliability diagrams for LogReg vs XGBoost on high and all (seed 0, before post-hoc calibration)."""
    d = np.load(RES / f"subset{BUDGET}_seed0" / "test_predictions.npz")
    y = d["y"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, fs in zip(axes, ("high", "all")):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=":", label="Perfect calibration")
        for model in ("logreg", "xgboost"):
            p = d[f"{model}|{fs}"]
            frac, mean = calibration_curve(y, p, n_bins=15)
            ece = expected_calibration_error(y, p)
            ax.plot(mean, frac, color=MODEL_COLORS[model], marker="o", ms=6, lw=2.5,
                    label=f"{LABEL[model]}  (ECE {ece:.3f})")
        ax.set_title(FS_LABEL[fs], loc="left")
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="upper left", fontsize=11.5)
    axes[0].set_ylabel("Observed signal fraction")
    fig.suptitle("XGBoost's probabilities can be trusted as-is; logistic regression's cannot",
                 x=0.01, ha="left", fontweight="bold", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "4_reliability.png")


def ece_before_after(main: pd.DataFrame) -> None:
    """ECE raw vs after Platt vs after isotonic, per model and feature set (mean over seeds)."""
    rows = [(m, fs) for m in ("logreg", "xgboost") for fs in FS_ORDER]
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(rows))[::-1]
    for (col, (name, color, marker)), dy in zip(METHOD.items(), (0.22, 0.0, -0.22)):
        vals = [main[(main.model == m) & (main.feature_set == fs)][f"{col}_mean"].iloc[0] for m, fs in rows]
        ax.scatter(vals, y + dy, s=110, color=color, marker=marker, edgecolor="#fcfcfb", linewidth=1.5, zorder=3,
                   label=name)
    ax.set_yticks(y, [f"{'LogReg' if m == 'logreg' else 'XGBoost'} · {fs}" for m, fs in rows])
    ax.axhline(len(FS_ORDER) - 0.5, color=GRID_LINE, lw=1)
    ax.set_xlabel("Expected calibration error (lower is better; mean over 5 seeds)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", title="Calibration")
    ax.annotate("Platt can't fix a mis-shaped curve;\nisotonic can", xy=(0.048, y[rows.index(("logreg", "high"))]),
                xytext=(0.03, y[rows.index(("logreg", "high"))] + 1.7), fontsize=12, color=INK,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    title(ax, "Post-hoc calibration, fit on the separate cal split",
          "XGBoost is already calibrated (~0.005). Logistic regression needs isotonic regression")
    save(fig, "5_calibration_before_after.png")


def auc_vs_time(main: pd.DataFrame) -> None:
    """Accuracy/compute trade-off at 250k: mean test AUC vs mean training time."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for model in ("logreg", "xgboost"):
        g = main[main.model == model].set_index("feature_set").loc[FS_ORDER]
        ax.scatter(g.train_seconds_mean, g.roc_auc_mean, s=140, color=MODEL_COLORS[model], edgecolor="#fcfcfb",
                   linewidth=2, zorder=3, label=LABEL[model])
        for fs, r in g.iterrows():
            secs = f"{r.train_seconds_mean:.2f}" if r.train_seconds_mean < 1 else f"{r.train_seconds_mean:.1f}"
            ax.annotate(f"{fs}  {secs}s", (r.train_seconds_mean, r.roc_auc_mean),
                        xytext=(10, -4), textcoords="offset points", fontsize=12, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 40)
    ax.set_xlabel("Training time in seconds (log scale; 200k training events, Apple M3)")
    ax.set_ylabel("Test ROC AUC")
    ax.legend(frameon=False, loc="lower right")
    title(ax, "Compute trade-off: XGBoost costs seconds and buys +0.14 to +0.17 AUC",
          "Best value: XGBoost on all, 4.8 s for 0.823. Best overall: XGBoost on geo + high, 0.838")
    save(fig, "6_auc_vs_training_time.png")


GRID_LINE = "#c3c2b7"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_df = pd.read_csv(RES / "summary" / f"main_{BUDGET}.csv")
    auc_by_features(main_df)
    significance_forest()
    tuning()
    reliability()
    ece_before_after(main_df)
    auc_vs_time(main_df)


if __name__ == "__main__":
    main()
