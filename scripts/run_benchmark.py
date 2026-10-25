"""Train every (model, feature set) pair on one seeded subset and evaluate on the shared test set.

Usage:
  python scripts/run_benchmark.py                          # everything in config.yaml
  python scripts/run_benchmark.py --subset-size 1000000 --models xgboost mlp --feature-sets low all
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
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.models import FITTERS  # noqa: E402  (first: imports xgboost before torch)
from higgs.data import ALL_FEATURE_SETS, load_splits  # noqa: E402
from higgs.metrics import evaluate  # noqa: E402

import torch  # noqa: E402
from higgs.plots import plot_auc_vs_time, plot_reliability, plot_roc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--subset-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--models", nargs="+", choices=list(FITTERS))
    ap.add_argument("--feature-sets", nargs="+", choices=ALL_FEATURE_SETS)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.subset_size:
        cfg["data"]["subset_size"] = args.subset_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    models = args.models or cfg["models"]
    feature_sets = args.feature_sets or cfg["feature_sets"]
    for key in ("array_path", "splits_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])

    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading subset ...", flush=True)
    splits = load_splits(cfg)
    m = splits.manifest
    print(f"  train {m['train']['n']:,}  val {m['val']['n']:,}  test {m['test']['n']:,}", flush=True)

    out_dir = ROOT / cfg["output_dir"] / f"subset{m['subset_size']}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "subset_manifest.json").write_text(json.dumps(m, indent=2))

    rows, preds, infos = [], {}, {}
    for fs in feature_sets:
        X_tr, X_val, X_te = splits.features(fs)
        scaler = StandardScaler().fit(X_tr)  # fitted on training rows only
        X_tr, X_val, X_te = (scaler.transform(a).astype(np.float32) for a in (X_tr, X_val, X_te))
        for model in models:
            print(f"[{model} | {fs}] training on {len(X_tr):,} rows x {X_tr.shape[1]} features", flush=True)
            t0 = time.perf_counter()
            fitted = FITTERS[model](X_tr, splits.y_train, X_val, splits.y_val, cfg[model], seed)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = fitted.predict_proba(X_te)
            infer_s = time.perf_counter() - t0
            metrics = evaluate(splits.y_test, p)
            print("    " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                  + f"  train {train_s:.1f}s", flush=True)
            rows.append({"model": model, "feature_set": fs, "n_features": X_tr.shape[1],
                         "subset_size": m["subset_size"], "seed": seed, **metrics,
                         "train_seconds": train_s, "inference_seconds": infer_s})
            preds[f"{model}|{fs}"] = p.astype(np.float32)
            infos[f"{model}|{fs}"] = fitted.info

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "metrics.csv", index=False)
    np.savez_compressed(out_dir / "test_predictions.npz", y=splits.y_test, **preds)
    (out_dir / "run_info.json").write_text(json.dumps({
        "models": infos, "python": platform.python_version(), "machine": platform.platform(),
        "processor": platform.processor(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
    }, indent=2, default=str))

    plot_roc(preds, splits.y_test, out_dir / "roc.png")
    plot_reliability(preds, splits.y_test, out_dir / "reliability.png")
    plot_auc_vs_time(results, out_dir / "auc_vs_train_time.png")

    print("\n" + results[["model", "feature_set", "roc_auc", "pr_auc", "brier", "ece", "train_seconds"]]
          .round(4).to_string(index=False))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
