"""Fast checks on synthetic data; no download required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from higgs.data import FEATURE_SETS, csv_to_array, feature_names, load_splits, make_split_indices  # noqa: E402
from higgs import data as data_mod  # noqa: E402
from higgs.metrics import evaluate, expected_calibration_error  # noqa: E402
from higgs.models import FITTERS  # noqa: E402


def test_feature_sets_partition_the_28_columns():
    assert len(FEATURE_SETS["low"]) == 21 and len(FEATURE_SETS["high"]) == 7
    assert FEATURE_SETS["low"] + FEATURE_SETS["high"] == FEATURE_SETS["all"]
    assert feature_names("high")[0] == "m_jj" and 0 not in FEATURE_SETS["all"]  # label excluded


def test_split_is_stratified_disjoint_and_seeded():
    rng = np.random.default_rng(0)
    labels = (rng.random(50_000) < 0.53).astype(np.int8)
    tr, va, ca = make_split_indices(labels, 10_000, 0.1, 0.1, seed=7)
    assert (len(tr), len(va), len(ca)) == (8_000, 1_000, 1_000)
    assert len(np.unique(np.concatenate([tr, va, ca]))) == 10_000          # pairwise disjoint
    for part in (tr, va, ca):
        assert abs(labels[part].mean() - labels.mean()) < 0.01           # stratified
    tr2, va2, ca2 = make_split_indices(labels, 10_000, 0.1, 0.1, seed=7)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(ca, ca2)
    tr3, _, _ = make_split_indices(labels, 10_000, 0.1, 0.1, seed=8)
    assert not np.array_equal(tr, tr3)


def test_csv_to_array_and_load_splits(tmp_path, monkeypatch):
    n = 2_000
    monkeypatch.setattr(data_mod, "N_ROWS", n)
    rng = np.random.default_rng(1)
    raw = np.column_stack([(rng.random(n) < 0.5).astype(np.float32), rng.normal(size=(n, 28))])
    # Same nesting as the UCI download: higgs.zip -> HIGGS.csv.gz -> CSV.
    import zipfile
    import pandas as pd
    gz = tmp_path / "HIGGS.csv.gz"
    pd.DataFrame(raw).to_csv(gz, header=False, index=False, compression="gzip")
    csv = tmp_path / "higgs.zip"
    with zipfile.ZipFile(csv, "w") as zf:
        zf.write(gz, "HIGGS.csv.gz")
    arr = csv_to_array(csv, tmp_path / "toy.npy", chunksize=300)
    np.testing.assert_allclose(np.load(arr), raw, rtol=1e-6)
    cfg = {"seed": 3, "data": {"array_path": str(arr), "splits_dir": str(tmp_path / "splits"),
                                "n_test": 500, "subset_size": 1_000, "val_fraction": 0.2, "cal_fraction": 0.1}}
    s = load_splits(cfg)
    assert s.X_train.shape == (700, 28) and s.X_val.shape == (200, 28) and s.X_cal.shape == (100, 28)
    assert s.X_test.shape == (500, 28)
    np.testing.assert_allclose(s.X_test, raw[-500:, 1:], rtol=1e-6)  # test = last rows, label removed
    np.testing.assert_array_equal(s.y_test, raw[-500:, 0])
    assert [a.shape for a in s.features("high")] == [(700, 7), (200, 7), (100, 7), (500, 7)]
    gh = s.features("geo_high")
    assert gh[0].shape == (700, 47)
    np.testing.assert_array_equal(gh[3][:, -7:], s.X_test[:, 21:])        # masses are the last 7 columns
    assert feature_names("geo_high")[-7:] == feature_names("high")
    assert (tmp_path / "splits" / "subset_1000_seed3_val0.2_cal0.1.json").exists()
    s2 = load_splits(cfg)  # reuses saved indices
    np.testing.assert_array_equal(s.X_train, s2.X_train)


def test_ece_is_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(2)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.15
    m = evaluate(y, p)
    assert set(m) == {"roc_auc", "pr_auc", "accuracy", "log_loss", "brier", "ece"}


def test_eda_quality_and_separation_checks():
    from higgs.eda import quality_report, row_overlap, single_feature_auc
    rng = np.random.default_rng(4)
    y = (rng.random(4_000) < 0.5).astype(np.int8)
    X = np.column_stack([rng.normal(size=4_000) + y, rng.normal(size=4_000), rng.integers(0, 3, 4_000)])
    X[1] = X[0]  # one duplicate row
    q = quality_report(X, y, ["informative", "noise", "btag_like"])
    assert q["duplicate_rows"] == 1 and q["nan_values"] == 0
    assert q["features"]["btag_like"]["discrete"] and not q["features"]["noise"]["discrete"]
    auc = single_feature_auc(X, y, ["informative", "noise", "btag_like"])
    assert auc.feature.iloc[0] == "informative" and auc.auc.iloc[0] > 0.7
    assert auc.set_index("feature").auc["noise"] < 0.55
    assert row_overlap(X[:100], X[50:200]) == 50


def _toy_events(n: int, seed: int) -> np.ndarray:
    """28 standardised columns with phi stored the way UCI stores it (radians / (pi / sqrt(3)))."""
    from higgs.physics import PHI_COLS
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 28)).astype(np.float32)
    X[:, PHI_COLS] = rng.uniform(-np.pi, np.pi, size=(n, len(PHI_COLS))) / (np.pi / np.sqrt(3))
    return X


def test_geo_features_shape_names_and_phi_scale():
    from higgs.physics import fit_phi_scale, geometric_features
    X = _toy_events(20_000, 5)
    scale = fit_phi_scale(X)
    assert abs(scale - np.pi / np.sqrt(3)) < 0.01
    G, names = geometric_features(X, scale)
    assert G.shape == (20_000, 40) and len(set(names)) == 40
    assert feature_names("geo") == names
    assert not any(n.endswith("_phi") for n in names)  # absolute angles dropped
    dphi = G[:, [i for i, n in enumerate(names) if n.startswith("dphi_")]]
    assert dphi.min() >= 0 and dphi.max() <= np.pi + 1e-5


def test_geo_features_are_invariant_to_rotation_about_the_beam():
    from higgs.physics import PHI_COLS, geometric_features
    X = _toy_events(5_000, 6)
    scale = np.pi / np.sqrt(3)
    rotated = X.copy()
    phi = rotated[:, PHI_COLS] * scale + 2.0          # rotate every object by the same angle
    rotated[:, PHI_COLS] = (((phi + np.pi) % (2 * np.pi)) - np.pi) / scale
    np.testing.assert_allclose(geometric_features(X, scale)[0], geometric_features(rotated, scale)[0], atol=2e-4)


def test_dphi_wraps_across_the_pi_boundary():
    from higgs.physics import _wrapped_abs_dphi
    got = _wrapped_abs_dphi(np.array([3.0, -3.0, 0.5]), np.array([-3.0, 3.0, -0.5]))
    np.testing.assert_allclose(got, [2 * np.pi - 6.0, 2 * np.pi - 6.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("model", list(FITTERS))
def test_models_learn_a_nonlinear_signal(model):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6_000, 5)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).astype(np.int8)
    params = {
        "logreg": {"C": 1.0, "max_iter": 500},
        "xgboost": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "subsample": 1.0,
                    "colsample_bytree": 1.0, "early_stopping_rounds": 20, "n_jobs": 2},
    }[model]
    fitted = FITTERS[model](X[:5_000], y[:5_000], X[5_000:], y[5_000:], params, seed=0)
    auc = evaluate(y[5_000:], fitted.predict_proba(X[5_000:]))["roc_auc"]
    assert auc > (0.75 if model == "logreg" else 0.9)


def test_delong_matches_sklearn_and_detects_a_real_difference():
    from sklearn.metrics import roc_auc_score
    from higgs.stats import auc_table, delong
    rng = np.random.default_rng(7)
    y = (rng.random(20_000) < 0.5).astype(int)
    strong = y + rng.normal(scale=1.0, size=y.size)
    weak = y + rng.normal(scale=2.0, size=y.size)
    aucs, cov = delong(np.vstack([strong, weak]), y)
    np.testing.assert_allclose(aucs, [roc_auc_score(y, strong), roc_auc_score(y, weak)], atol=1e-9)
    assert cov.shape == (2, 2) and np.all(np.diag(cov) > 0)
    # Variance agrees with a bootstrap estimate to within ~20%.
    boot = [roc_auc_score(y[i], strong[i]) for i in (rng.integers(0, y.size, y.size) for _ in range(200))]
    assert 0.8 < np.var(boot) / cov[0, 0] < 1.25
    ci, sig = auc_table({"strong": strong, "weak": weak, "strong_copy": strong.copy()}, y)
    pair = sig.set_index(["a", "b"])
    assert pair.loc[("strong", "weak"), "p_value"] < 1e-6
    assert pair.loc[("strong", "strong_copy"), "diff"] == 0
    assert (ci.auc_lo < ci.auc).all() and (ci.auc < ci.auc_hi).all()


def test_calibrators_fix_a_miscalibrated_model_without_changing_ranking():
    from sklearn.metrics import roc_auc_score
    from higgs.calibration import CALIBRATORS
    rng = np.random.default_rng(8)
    p_true = rng.random(40_000)
    y = (rng.random(40_000) < p_true).astype(int)
    p_bad = p_true ** 3                                   # monotone distortion -> badly calibrated
    raw_ece = expected_calibration_error(y[20_000:], p_bad[20_000:])
    for name, Cal in CALIBRATORS.items():
        fixed = Cal().fit(p_bad[:20_000], y[:20_000]).predict(p_bad[20_000:])
        assert expected_calibration_error(y[20_000:], fixed) < raw_ece / 3, name
        if name == "platt":
            assert abs(roc_auc_score(y[20_000:], fixed) - roc_auc_score(y[20_000:], p_bad[20_000:])) < 1e-5  # clipping near 0 adds a few ties
