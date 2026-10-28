"""Train every (model, feature set) pair on one seeded subset and evaluate on the shared test set.

Per feature set: fit the scaler on train; fit each model (XGBoost early-stops on val); fit post-hoc
calibrators on cal; score everything on the 500k test events; then DeLong CIs for every AUC and
paired DeLong tests for every pair of predictions.

Usage:
  python scripts/run_benchmark.py                                   # everything in config.yaml
  python scripts/run_benchmark.py --seed 1 --subset-size 1000000 --models xgboost --feature-sets low all
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.calibration import CALIBRATORS  # noqa: E402
from higgs.data import ALL_FEATURE_SETS, load_splits  # noqa: E402
from higgs.metrics import evaluate  # noqa: E402
from higgs.models import FITTERS  # noqa: E402
from higgs.plots import plot_auc_vs_time, plot_reliability, plot_roc  # noqa: E402
from higgs.stats import auc_table  # noqa: E402


def load_config(args: argparse.Namespace) -> dict:
    """config.yaml plus CLI overrides; paths made absolute."""
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.subset_size:
        cfg["data"]["subset_size"] = args.subset_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.models:
        cfg["models"] = args.models
    if args.feature_sets:
        cfg["feature_sets"] = args.feature_sets
    for key in ("array_path", "splits_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    return cfg


def params_for(cfg: dict, tuned: dict, model: str, fs: str) -> dict:
    """Default hyperparameters for `model`, overridden by CV-tuned values for this feature set."""
    return {**cfg[model], **tuned.get(model, {}).get(fs, {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--subset-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--models", nargs="+", choices=list(FITTERS))
    ap.add_argument("--feature-sets", nargs="+", choices=ALL_FEATURE_SETS)
    args = ap.parse_args()
    cfg = load_config(args)

    seed = int(cfg["seed"])
    np.random.seed(seed)
    tuned_path = ROOT / cfg.get("tuned_params", "")
    tuned = json.loads(tuned_path.read_text()) if tuned_path.is_file() else {}
    print(f"Tuned hyperparameters: {'loaded from ' + str(tuned_path) if tuned else 'none (defaults)'}", flush=True)

    print("Loading subset ...", flush=True)
    splits = load_splits(cfg)
    m = splits.manifest
    print(f"  train {m['train']['n']:,}  val {m['val']['n']:,}  cal {m['cal']['n']:,}  test {m['test']['n']:,}",
          flush=True)

    out_dir = ROOT / cfg["output_dir"] / f"subset{m['subset_size']}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "subset_manifest.json").write_text(json.dumps(m, indent=2))

    y_cal, y_te = splits.y_cal, splits.y_test
    rows, preds, infos = [], {}, {}
    for fs in cfg["feature_sets"]:
        X_tr, X_val, X_cal, X_te = splits.features(fs)
        scaler = StandardScaler().fit(X_tr)  # fitted on training rows only
        X_tr, X_val, X_cal, X_te = (scaler.transform(a).astype(np.float32, copy=False)
                                    for a in (X_tr, X_val, X_cal, X_te))
        for model in cfg["models"]:
            params = params_for(cfg, tuned, model, fs)
            print(f"[{model} | {fs}] training on {len(X_tr):,} rows x {X_tr.shape[1]} features", flush=True)
            t0 = time.perf_counter()
            fitted = FITTERS[model](X_tr, splits.y_train, X_val, splits.y_val, params, seed)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p_te = fitted.predict_proba(X_te)
            infer_s = time.perf_counter() - t0
            p_cal = fitted.predict_proba(X_cal)
            row = {"model": model, "feature_set": fs, "n_features": X_tr.shape[1],
                   "subset_size": m["subset_size"], "n_train": len(X_tr), "seed": seed,
                   **evaluate(y_te, p_te), "train_seconds": train_s, "inference_seconds": infer_s}
            for name, Cal in CALIBRATORS.items():   # post-hoc calibration, fit on cal only
                cm = evaluate(y_te, Cal().fit(p_cal, y_cal).predict(p_te))
                row.update({f"{k}_{name}": cm[k] for k in ("ece", "brier", "log_loss")})
            rows.append(row)
            print("    " + "  ".join(f"{k} {row[k]:.4f}" for k in ("roc_auc", "pr_auc", "ece", "ece_platt"))
                  + f"  train {train_s:.1f}s", flush=True)
            preds[f"{model}|{fs}"] = p_te.astype(np.float32)
            infos[f"{model}|{fs}"] = {"params": params, **fitted.info}
        del X_tr, X_val, X_cal, X_te

    print("DeLong confidence intervals and paired tests ...", flush=True)
    ci, sig = auc_table(preds, y_te)
    results = pd.DataFrame(rows)
    results["key"] = results.model + "|" + results.feature_set
    results = results.merge(ci.drop(columns="auc"), on="key", how="left").drop(columns="key")
    results.to_csv(out_dir / "metrics.csv", index=False)
    sig.to_csv(out_dir / "significance.csv", index=False)
    np.savez_compressed(out_dir / "test_predictions.npz", y=y_te, **preds)
    (out_dir / "run_info.json").write_text(json.dumps({
        "models": infos, "python": platform.python_version(), "machine": platform.platform(),
        "processor": platform.processor(), "xgboost": xgboost.__version__,
    }, indent=2, default=str))

    plot_roc(preds, y_te, out_dir / "roc.png")
    plot_reliability(preds, y_te, out_dir / "reliability.png", n_bins=cfg.get("calibration_bins", 15))
    plot_auc_vs_time(results, out_dir / "auc_vs_train_time.png")

    print("\n" + results[["model", "feature_set", "roc_auc", "auc_lo", "auc_hi", "pr_auc", "ece", "ece_platt",
                          "train_seconds"]].round(4).to_string(index=False))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
