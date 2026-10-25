"""MLP ablation study: change one design choice at a time against the baseline, over several seeds.

Baseline = config.yaml's `mlp` block (3x256, ReLU, BatchNorm, dropout 0.1, Glorot init, wd 1e-5).
The tanh / no-BatchNorm variants reproduce the setting Glorot & Bengio (2010) studied, where the
initialisation matters most; with BatchNorm the network should be largely insensitive to it.

Usage: python scripts/run_ablations.py [--seeds 0 1 2] [--feature-sets low all] [--subset-size 250000]
Writes results/ablations/{metrics.csv, histories.json, ablation_auc.png, learning_curves.png, summary.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.models import fit_mlp  # noqa: E402  (first: imports xgboost before torch)
from higgs.data import ALL_FEATURE_SETS, load_splits  # noqa: E402
from higgs.metrics import evaluate  # noqa: E402
from higgs.plots import plot_ablation_bars, plot_learning_curves  # noqa: E402

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "no_batchnorm": {"batchnorm": False},
    "no_dropout": {"dropout": 0.0},
    "dropout_0.3": {"dropout": 0.3},
    "no_weight_decay": {"weight_decay": 0.0},
    "default_init": {"init": "default"},
    "default_init_no_bn": {"init": "default", "batchnorm": False},
    "tanh_no_bn_xavier": {"activation": "tanh", "batchnorm": False},
    "tanh_no_bn_default_init": {"activation": "tanh", "batchnorm": False, "init": "default"},
    "depth_1": {"hidden": [256]},
    "depth_5": {"hidden": [256] * 5},
    "width_64": {"hidden": [64] * 3},
    "width_512": {"hidden": [512] * 3},
}
CURVE_VARIANTS = ["baseline", "no_batchnorm", "no_dropout", "tanh_no_bn_xavier", "tanh_no_bn_default_init"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--feature-sets", nargs="+", default=["low", "all"], choices=ALL_FEATURE_SETS)
    ap.add_argument("--subset-size", type=int)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    if args.subset_size:
        cfg["data"]["subset_size"] = args.subset_size
    for key in ("array_path", "splits_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    out = ROOT / "results" / "ablations"
    out.mkdir(parents=True, exist_ok=True)
    variants = args.variants or list(VARIANTS)

    rows, histories = [], {}
    for seed in args.seeds:
        cfg["seed"] = seed
        splits = load_splits(cfg)
        for fs in args.feature_sets:
            X_tr, X_val, _, X_te = splits.features(fs)
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_val, X_te = (scaler.transform(a).astype(np.float32, copy=False) for a in (X_tr, X_val, X_te))
            for name in variants:
                params = {**cfg["mlp"], **VARIANTS[name]}
                t0 = time.perf_counter()
                model = fit_mlp(X_tr, splits.y_train, X_val, splits.y_val, params, seed, verbose=False)
                secs = time.perf_counter() - t0
                met = evaluate(splits.y_test, model.predict_proba(X_te))
                rows.append({"variant": name, "feature_set": fs, "seed": seed, **met, "train_seconds": secs,
                             "best_epoch": model.info["best_epoch"], "epochs_run": model.info["epochs_run"],
                             "n_params": model.info["n_params"]})
                histories[f"{name}|{fs}|{seed}"] = model.info["history"]
                print(f"[seed {seed} | {fs} | {name}] auc {met['roc_auc']:.4f} ece {met['ece']:.4f} "
                      f"epochs {model.info['epochs_run']} {secs:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)   # checkpoint as we go

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    (out / "histories.json").write_text(json.dumps(histories))

    base = df[df.variant == "baseline"].set_index(["feature_set", "seed"]).roc_auc
    df["delta_auc"] = df.roc_auc - df.set_index(["feature_set", "seed"]).index.map(base).to_numpy()
    summary = (df.groupby(["feature_set", "variant"], sort=False)
                 .agg(auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"),
                      delta_mean=("delta_auc", "mean"), delta_std=("delta_auc", "std"),
                      ece_mean=("ece", "mean"), epochs=("epochs_run", "mean"),
                      n_params=("n_params", "first"), seconds=("train_seconds", "mean"))
                 .reset_index())
    summary.to_csv(out / "summary.csv", index=False)
    plot_ablation_bars(summary, out / "ablation_auc.png")
    plot_learning_curves(histories, [v for v in CURVE_VARIANTS if v in variants],
                         args.feature_sets[0], args.seeds[0], out / "learning_curves.png")

    lines = [f"# MLP ablations ({len(args.seeds)} seeds: {args.seeds}; budget {splits.manifest['subset_size']:,})", ""]
    for fs, g in summary.groupby("feature_set", sort=False):
        lines += [f"## `{fs}` features", "", "| variant | test AUC (mean ± sd) | Δ vs baseline | ECE | epochs | params | s |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r.variant} | {r.auc_mean:.4f} ± {r.auc_std:.4f} | {r.delta_mean:+.4f} ± {r.delta_std:.4f} | "
                  f"{r.ece_mean:.4f} | {r.epochs:.0f} | {r.n_params:,} | {r.seconds:.0f} |" for r in g.itertuples()]
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
