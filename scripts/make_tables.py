"""Generate the report's LaTeX tables and copy its figures from results/ (no hand-typed numbers).

Writes report/tables/*.tex and report/figures/*.png. Run after scripts/aggregate.py.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
BUDGET = 250000
MODEL_TEX = {"logreg": "LogReg", "xgboost": "XGBoost"}
FS_TEX = {"low": "low (21)", "geo": "geo (40)", "high": "high (7)", "all": "all (28)", "geo_high": "geo+high (47)"}
FS_ORDER = list(FS_TEX)


def pm(mean: float, sd: float, d: int = 4) -> str:
    return f"{mean:.{d}f}" if pd.isna(sd) else f"{mean:.{d}f}$\\pm${sd:.{d}f}"


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&")


def table(name: str, caption: str, label: str, header: list[str], rows: list[list[str]], wide: bool = False,
          align: str | None = None) -> None:
    env = "table*" if wide else "table"
    align = align or "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote tables/{name}.tex")


def main_rows(main: pd.DataFrame, models: list[str]) -> list[list[str]]:
    rows = []
    for fs in FS_ORDER:
        for m in models:
            r = main[(main.model == m) & (main.feature_set == fs)]
            if r.empty:
                continue
            r = r.iloc[0]
            best = r.roc_auc_mean == main[main.feature_set == fs].roc_auc_mean.max()
            auc = pm(r.roc_auc_mean, r.roc_auc_std)
            rows.append([FS_TEX[fs], MODEL_TEX[m], f"\\textbf{{{auc}}}" if best else auc,
                         pm(r.pr_auc_mean, r.pr_auc_std), pm(r.brier_mean, r.brier_std),
                         f"{r.ece_mean:.4f}", "--" if pd.isna(r.ece_platt_mean) else f"{r.ece_platt_mean:.4f}",
                         "--" if pd.isna(r.ece_isotonic_mean) else f"{r.ece_isotonic_mean:.4f}",
                         f"{r.train_seconds_mean:.1f}"])
    return rows


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    head = ["Features", "Model", "ROC AUC", "PR AUC", "Brier", "ECE", "ECE$_\\mathrm{Platt}$",
            "ECE$_\\mathrm{iso}$", "Train s"]

    main = pd.read_csv(RES / "summary" / f"main_{BUDGET}.csv")
    n_seeds = int(main.n_seeds.max())
    table("ml_main", f"Results at a {BUDGET:,}-event budget: mean$\\pm$sd over {n_seeds} seeds, "
          "all scored on the same 500k test events. ECE before and after post-hoc calibration on the cal split. "
          "Bold: best ROC AUC within the feature set.",
          "tab:ml_main", head, main_rows(main, ["logreg", "xgboost"]), wide=True)

    comp = pd.read_csv(RES / "summary" / f"comparisons_{BUDGET}.csv")
    table("comparisons", "Paired DeLong tests on the shared test set. $\\Delta$AUC = A$-$B (mean$\\pm$sd over "
          "seeds); last column: seeds in which the difference is significant at $\\alpha=0.05$ in favour of A / B.",
          "tab:comparisons", ["Question", "A", "B", "$\\Delta$AUC", "Sig. A/B"],
          [[esc(r.question), esc(r.a), esc(r.b), pm(r.diff_mean, r.diff_sd), f"{r.seeds_a_better_p05}/{r.seeds_b_better_p05}"]
           for r in comp.itertuples()], wide=True, align="lllcc")

    best = json.loads((RES / "tuning" / "best_params.json").read_text())
    cv = pd.read_csv(RES / "tuning" / "cv_results.csv")
    rows = []
    for fs in FS_ORDER:
        if fs not in best["xgboost"]:
            continue
        x = best["xgboost"][fs]
        cx = cv[(cv.model == "xgboost") & (cv.feature_set == fs)].cv_auc.max()
        cl = cv[(cv.model == "logreg") & (cv.feature_set == fs)].cv_auc.max()
        rows.append([FS_TEX[fs], f"{best['logreg'][fs]['C']:g}", f"{cl:.4f}", str(x["max_depth"]), f"{x['learning_rate']:g}",
                     str(x["min_child_weight"]), f"{x['subsample']:g}", f"{x['colsample_bytree']:g}",
                     f"{x['reg_lambda']:g}", f"{cx:.4f}"])
    table("tuning", "Hyperparameters selected by stratified 3-fold CV on the training split (test untouched).",
          "tab:tuning", ["Features", "C", "CV AUC", "depth", "$\\eta$", "mcw", "subs.", "colsm.", "$\\lambda$", "CV AUC"],
          rows, wide=True)

    sc = pd.read_csv(RES / "summary" / "scaling.csv")
    budgets = sorted(sc.subset_size.unique())
    for name, models, fsets in (("scaling_ml", ("logreg", "xgboost"), FS_ORDER),):
        rows = []
        for fs in fsets:
            for m in models:
                g = sc[(sc.feature_set == fs) & (sc.model == m)].set_index("subset_size")
                if g.empty:
                    continue
                rows.append([FS_TEX[fs], MODEL_TEX[m]] + [pm(g.auc_mean[b], g.auc_sd[b]) if b in g.index else "--"
                                                         for b in budgets])
        table(name, "Test ROC AUC vs data budget (train+val+cal events); mean$\\pm$sd over seeds "
              "(3 seeds; 5 at 250k; 1 at 10.5M, \\texttt{low}/\\texttt{all} only).", f"tab:{name}",
              ["Features", "Model"] + [f"{b / 1e6:g}M" if b >= 1e6 else f"{b // 1000}k" for b in budgets],
              rows, wide=True)

    imp = pd.read_csv(RES / "interpretability" / "importance_all.csv").head(10)
    table("importance_ml", "Top-10 features of the \\texttt{all} set by XGBoost mean $|$SHAP$|$ (log-odds), with "
          "logistic-regression $|$coefficients$|$ on standardised inputs.", "tab:importance_ml",
          ["Feature", "XGB $|$SHAP$|$", "LogReg $|\\beta|$"],
          [[esc(r.feature), f"{r.xgb_mean_abs_shap:.3f}", f"{r.logreg_abs_coef:.3f}"] for r in imp.itertuples()])

    eda_dir = next((RES / "eda").glob("subset*_seed0"))
    q = json.loads((eda_dir / "quality_report.json").read_text())
    sfa = pd.read_csv(eda_dir / "single_feature_auc.csv").head(8)
    table("single_auc", "Most separating single features (training split): ROC AUC of the feature alone.",
          "tab:single_auc", ["Feature", "Group", "AUC"], [[esc(r.feature), r.group, f"{r.auc:.3f}"] for r in sfa.itertuples()])
    (TAB / "eda_numbers.tex").write_text(
        f"\\newcommand{{\\edaN}}{{{q['n_events']:,}}}\n" +
        f"\\newcommand{{\\edaSignal}}{{{q['signal_fraction']:.3f}}}\n"
        f"\\newcommand{{\\edaDup}}{{{q['duplicate_rows']}}}\n"
        f"\\newcommand{{\\edaOverlap}}{{{q['test_rows_also_in_train']}}}\n")

    figures = {
        "eda_distributions.png": eda_dir / "distributions.png",
        "eda_correlation.png": eda_dir / "correlation.png",
        "eda_single_auc.png": eda_dir / "single_feature_auc.png",
        "reliability_ml.png": ROOT / "presentation" / "phase1_ml" / "figures" / "4_reliability.png",
        "shap_dependence.png": RES / "interpretability" / "shap_dependence_all.png",
    }
    for dst, src in figures.items():
        if src.exists():
            shutil.copy(src, FIG / dst)
            print(f"  copied figures/{dst}")
        else:
            print(f"  MISSING {src}")


if __name__ == "__main__":
    main()
