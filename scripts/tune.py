"""Hyperparameter selection by stratified 3-fold CV on the TRAINING split only (test never touched).

Logistic regression: grid over C (scaler inside each fold via a Pipeline).
XGBoost: seeded random search; each fold early-stops on its own held-out fold, so every configuration
gets the same mild optimism and the ranking is fair. Selection metric: mean CV ROC AUC.
Writes results/tuning/best_params.json (read by run_benchmark.py) and cv_results.csv.

Usage: python scripts/tune.py [--seed 0] [--n-iter 12] [--feature-sets low geo ...]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.models import make_xgb  # noqa: E402
from higgs.data import ALL_FEATURE_SETS, load_splits  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

LOGREG_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
XGB_SPACE = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}


def tune_logreg(X, y, folds, seed: int) -> tuple[dict, list[dict]]:
    """Grid over the inverse regularisation strength C."""
    rows = []
    for C in LOGREG_GRID:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=seed))
        scores = cross_val_score(pipe, X, y, cv=folds, scoring="roc_auc")
        rows.append({"model": "logreg", "params": {"C": C}, "cv_auc": scores.mean(), "cv_auc_std": scores.std()})
    best = max(rows, key=lambda r: r["cv_auc"])
    return best["params"], rows


def tune_xgboost(X, y, folds, base: dict, n_iter: int, seed: int) -> tuple[dict, list[dict]]:
    """Seeded random search; per-fold early stopping on the held-out fold."""
    rows = []
    for cand in ParameterSampler(XGB_SPACE, n_iter=n_iter, random_state=seed):
        params = {**base, **cand}
        aucs, iters, t0 = [], [], time.perf_counter()
        for tr, va in folds.split(X, y):
            est = make_xgb(params, seed).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            aucs.append(roc_auc_score(y[va], est.predict_proba(X[va])[:, 1]))
            iters.append(est.best_iteration)
        rows.append({"model": "xgboost", "params": cand, "cv_auc": float(np.mean(aucs)),
                     "cv_auc_std": float(np.std(aucs)), "mean_best_iteration": float(np.mean(iters)),
                     "seconds": time.perf_counter() - t0})
        print(f"    xgb {cand} -> cv_auc {rows[-1]['cv_auc']:.4f}", flush=True)
    best = max(rows, key=lambda r: r["cv_auc"])
    return best["params"], rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--feature-sets", nargs="+", choices=ALL_FEATURE_SETS)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["seed"] = args.seed
    for key in ("array_path", "splits_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    splits = load_splits(cfg)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    out = ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)

    best: dict = {"logreg": {}, "xgboost": {}}
    all_rows = []
    for fs in args.feature_sets or cfg["feature_sets"]:
        X = splits.features(fs)[0]
        y = splits.y_train
        print(f"[{fs}] tuning on {len(y):,} training rows x {X.shape[1]} features", flush=True)
        best["logreg"][fs], rows = tune_logreg(X, y, folds, args.seed)
        all_rows += [{**r, "feature_set": fs} for r in rows]
        print(f"    logreg best {best['logreg'][fs]}", flush=True)
        best["xgboost"][fs], rows = tune_xgboost(X, y, folds, cfg["xgboost"], args.n_iter, args.seed)
        all_rows += [{**r, "feature_set": fs} for r in rows]
        print(f"    xgboost best {best['xgboost'][fs]}", flush=True)

    (out / "best_params.json").write_text(json.dumps(best, indent=2))
    df = pd.DataFrame(all_rows)
    df["params"] = df.params.map(json.dumps)
    df.sort_values(["feature_set", "model", "cv_auc"], ascending=[True, True, False]).to_csv(
        out / "cv_results.csv", index=False)
    print(f"Wrote {out / 'best_params.json'}")


if __name__ == "__main__":
    main()
