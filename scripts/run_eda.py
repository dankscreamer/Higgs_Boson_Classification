"""Data-quality report and EDA on the training split of the configured subset (test set untouched).

Usage: python scripts/run_eda.py [--subset-size N] [--seed S]
Writes results/eda/subset<N>_seed<S>/: quality_report.json, single_feature_auc.csv, mass_peaks.csv,
summary.md and figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.data import FEATURE_NAMES, HIGH_LEVEL, load_splits  # noqa: E402
from higgs.eda import (  # noqa: E402
    mass_peaks, plot_correlation, plot_distributions, plot_single_feature_auc, quality_report,
    row_overlap, single_feature_auc,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--subset-size", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.subset_size:
        cfg["data"]["subset_size"] = args.subset_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    for key in ("array_path", "splits_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])

    splits = load_splits(cfg)
    X, y, names = splits.X_train, splits.y_train, FEATURE_NAMES
    m = splits.manifest
    out = ROOT / cfg["output_dir"] / "eda" / f"subset{m['subset_size']}_seed{m['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"EDA on {len(y):,} training events -> {out}", flush=True)

    q = quality_report(X, y, names)
    # Only an exact-match count touches the test set; no test statistics are looked at.
    q["test_rows_also_in_train"] = row_overlap(X, splits.X_test)
    (out / "quality_report.json").write_text(json.dumps(q, indent=2))
    auc = single_feature_auc(X, y, names)
    auc.to_csv(out / "single_feature_auc.csv", index=False)
    peaks = mass_peaks(X, y, names)
    peaks.to_csv(out / "mass_peaks.csv", index=False)
    corr = plot_correlation(X, names, out / "correlation.png")
    plot_distributions(X, y, names, list(auc.feature), out / "distributions.png")
    plot_single_feature_auc(auc, out / "single_feature_auc.png")

    # Physics sanity checks, stated as expectations and observed values.
    phi = auc[auc.feature.str.endswith("_phi")]
    btag = {n: q["features"][n]["n_unique"] for n in names if n.endswith("_btag")}
    low_names = [n for n in names if n not in HIGH_LEVEL]
    cross = corr.loc[HIGH_LEVEL, low_names].abs()
    top_cross = cross.stack().sort_values(ascending=False).head(5)
    narrow = peaks.pivot(index="feature", columns="class", values="iqr")
    narrow["signal_narrower"] = narrow.signal < narrow.background

    lines = [
        f"# EDA summary: subset {m['subset_size']:,}, seed {m['seed']} (training split only)",
        "",
        "## Data quality",
        f"- Events: {q['n_events']:,}; signal fraction {q['signal_fraction']:.3f} (mildly imbalanced, so we report PR AUC as well as accuracy)",
        f"- NaN values: {q['nan_values']}; infinite values: {q['inf_values']}; duplicate rows: {q['duplicate_rows']}",
        f"- Test events that also appear exactly in train: {q['test_rows_also_in_train']} of {len(splits.y_test):,} "
        "(the UCI file contains duplicate events; this is too few to move the metrics, and we keep the standard "
        "test set so results stay comparable with Baldi et al.)",
        f"- Constant features: {q['constant_features'] or 'none'}",
        f"- Discrete features: {[n for n, s in q['features'].items() if s['discrete']]}",
        "- Units: features are pre-scaled in the UCI release (for example, momenta have a mean near 1), so masses are not in GeV.",
        "",
        "## Physics checks",
        f"- **phi is uninformative alone** (rotational symmetry about the beam): single-feature AUC "
        f"{phi.auc.min():.3f} to {phi.auc.max():.3f}.",
        f"- **b-tag is discrete**: number of distinct values per jet {btag}.",
        "- **Mass peaks**: interquartile range of each invariant mass, signal vs background "
        "(narrower = more peaked):",
        "",
        "| mass | signal IQR | background IQR | signal narrower? |",
        "|---|---|---|---|",
        *[f"| {f} | {r.signal:.3f} | {r.background:.3f} | {'yes' if r.signal_narrower else 'no'} |"
          for f, r in narrow.loc[HIGH_LEVEL].iterrows()],
        "",
        "- **Expert features are functions of the raw ones**: strongest |correlation| between expert and raw features: "
        + ", ".join(f"{a}~{b} {v:.2f}" for (a, b), v in top_cross.items()),
        "",
        "## Single-feature separating power (ROC AUC, 0.5 = none)",
        "",
        "| rank | feature | group | AUC |",
        "|---|---|---|---|",
        *[f"| {i + 1} | {r.feature} | {r.group} | {r.auc:.3f} |" for i, r in auc.head(10).iterrows()],
        "",
        f"The best single feature reaches AUC {auc.auc.iloc[0]:.3f}, while models that combine features reach above 0.8 "
        "(see the benchmark). Separating these events needs combinations of features, which is the case for ML.",
        "",
        "These are simulated events. Results describe separability on this benchmark, not a physical discovery.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
