"""What do the models rely on? Three complementary importance measures on the same features.

- Logistic regression: |coefficient| on standardised inputs (global, linear).
- XGBoost: mean |TreeSHAP| contribution (Lundberg et al. 2020), computed exactly by xgboost itself
  (`pred_contribs=True`), so no extra dependency; plus SHAP dependence plots for key masses.
- MLP: permutation importance = drop in test ROC AUC when one feature is shuffled (model-agnostic).
All importances are computed on a fixed 50k-event sample of the test set.

Usage: python scripts/interpret.py [--seed 0] [--feature-sets all geo]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higgs.models import FITTERS  # noqa: E402  (first: imports xgboost before torch)
from xgboost import DMatrix  # noqa: E402
from higgs.data import ALL_FEATURE_SETS, feature_names, load_splits  # noqa: E402
from higgs.plots import plot_importance, plot_shap_dependence  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def permutation_importance(model, X: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    """Mean drop in ROC AUC when each column is shuffled (columns shuffled independently)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X))
    drops = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[j] += base - roc_auc_score(y, model.predict_proba(Xp))
    return drops / repeats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-sets", nargs="+", default=["all", "geo"], choices=ALL_FEATURE_SETS)
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["seed"] = args.seed
    for key in ("array_path", "splits_dir"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])
    tuned_path = ROOT / cfg["tuned_params"]
    tuned = json.loads(tuned_path.read_text()) if tuned_path.is_file() else {}
    splits = load_splits(cfg)
    out = ROOT / "results" / "interpretability"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(len(splits.y_test), size=args.sample, replace=False))
    y_s = splits.y_test[sample]

    for fs in args.feature_sets:
        names = feature_names(fs)
        X_tr, X_val, _, X_te = splits.features(fs)
        raw_sample = X_te[sample]
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_val, X_te = (scaler.transform(a).astype(np.float32, copy=False) for a in (X_tr, X_val, X_te))
        X_s = X_te[sample]
        models = {}
        for m in ("logreg", "xgboost", "mlp"):
            params = {**cfg[m], **tuned.get(m, {}).get(fs, {})}
            print(f"[{fs}] fitting {m}", flush=True)
            models[m] = FITTERS[m](X_tr, splits.y_train, X_val, splits.y_val, params, args.seed)

        coef = np.abs(models["logreg"].est.coef_[0])
        contribs = models["xgboost"].est.get_booster().predict(DMatrix(X_s), pred_contribs=True)[:, :-1]
        shap_mean = np.abs(contribs).mean(axis=0)
        print(f"[{fs}] permutation importance for the MLP ({len(names)} features x {args.repeats})", flush=True)
        perm = permutation_importance(models["mlp"], X_s, y_s, args.repeats, args.seed)

        imp = pd.DataFrame({"feature": names, "logreg_abs_coef": coef, "xgb_mean_abs_shap": shap_mean,
                            "mlp_perm_auc_drop": perm}).sort_values("xgb_mean_abs_shap", ascending=False)
        imp.to_csv(out / f"importance_{fs}.csv", index=False)
        plot_importance(imp, out / f"importance_{fs}.png", title=f"{fs} features")
        dep = [n for n in ("m_bb", "m_wwbb", "m_jjj") if n in names]
        if dep:
            plot_shap_dependence({n: (raw_sample[:, names.index(n)], contribs[:, names.index(n)]) for n in dep},
                                 out / f"shap_dependence_{fs}.png")
        print(imp.head(10).round(4).to_string(index=False), flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
