"""
Unit tests for higgs.data module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from higgs.data import (
    ALL_FEATURES,
    COLUMN_NAMES,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
    check_class_balance,
    check_constant_features,
    check_duplicates,
    check_missing_and_infinite_values,
    load_higgs_sample,
    split_dataset,
    validate_schema,
    validate_target_labels,
)


@pytest.fixture
def synthetic_higgs_df() -> pd.DataFrame:
    """Creates a small synthetic DataFrame strictly adhering to HIGGS schema."""
    rng = np.random.default_rng(42)
    n_samples = 200

    data = {TARGET_COLUMN: rng.choice([0.0, 1.0], size=n_samples, p=[0.47, 0.53])}
    for feat in ALL_FEATURES:
        data[feat] = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    return pd.DataFrame(data)


def test_schema_constants():
    """Verify feature counts and column structure."""
    assert len(LOW_LEVEL_FEATURES) == 21
    assert len(HIGH_LEVEL_FEATURES) == 7
    assert len(ALL_FEATURES) == 28
    assert len(COLUMN_NAMES) == 29
    assert COLUMN_NAMES[0] == TARGET_COLUMN
    assert COLUMN_NAMES[1:] == ALL_FEATURES


def test_validate_schema_success(synthetic_higgs_df):
    """Verify validate_schema passes on valid schema."""
    report = validate_schema(synthetic_higgs_df)
    assert report["column_count_matches"] is True
    assert report["column_names_match"] is True
    assert report["all_numeric"] is True


def test_validate_schema_wrong_col_count():
    """Verify validate_schema raises on invalid column count."""
    df_invalid = pd.DataFrame({"class_label": [0.0, 1.0], "lepton_pT": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_schema(df_invalid)


def test_validate_target_labels_success(synthetic_higgs_df):
    """Verify validate_target_labels passes for valid binary target."""
    report = validate_target_labels(synthetic_higgs_df)
    assert report["is_binary"] is True
    assert report["unique_values"] == [0.0, 1.0]
    assert report["missing_target_count"] == 0


def test_validate_target_labels_invalid():
    """Verify validate_target_labels catches non-binary values."""
    df = pd.DataFrame({TARGET_COLUMN: [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="invalid non-binary values"):
        validate_target_labels(df)


def test_check_class_balance(synthetic_higgs_df):
    """Verify class balance calculation."""
    summary = check_class_balance(synthetic_higgs_df)
    assert "Class" in summary.columns
    assert "Count" in summary.columns
    assert "Percentage (%)" in summary.columns
    assert summary["Count"].sum() == len(synthetic_higgs_df)


def test_check_missing_and_infinite_values(synthetic_higgs_df):
    """Verify missing and infinite value detection."""
    df_dirty = synthetic_higgs_df.copy()
    df_dirty.loc[0, "lepton_pT"] = np.nan
    df_dirty.loc[1, "m_bb"] = np.inf
    df_dirty.loc[2, "m_wwbb"] = -np.inf

    report = check_missing_and_infinite_values(df_dirty)
    assert report.loc["lepton_pT", "Missing (NaN)"] == 1
    assert report.loc["m_bb", "+Infinity"] == 1
    assert report.loc["m_wwbb", "-Infinity"] == 1
    assert report.loc["lepton_eta", "Total Anomalies"] == 0


def test_check_constant_features(synthetic_higgs_df):
    """Verify constant feature detection."""
    df_const = synthetic_higgs_df.copy()
    df_const["lepton_pT"] = 42.0  # Zero variance

    report = check_constant_features(df_const)
    pt_row = report[report["Feature"] == "lepton_pT"].iloc[0]
    assert bool(pt_row["Is Constant"]) is True
    assert pt_row["Variance"] == 0.0

    eta_row = report[report["Feature"] == "lepton_eta"].iloc[0]
    assert bool(eta_row["Is Constant"]) is False


def test_check_duplicates(synthetic_higgs_df):
    """Verify duplicate detection."""
    df_dup = pd.concat([synthetic_higgs_df, synthetic_higgs_df.iloc[:5]], ignore_index=True)
    report = check_duplicates(df_dup)
    assert report["duplicate_rows"] == 5
    assert report["total_rows_inspected"] == len(df_dup)


def test_split_dataset_proportions_and_stratification(synthetic_higgs_df):
    """Verify reproducible splitting, proportions, and stratification."""
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
        synthetic_higgs_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    total_rows = len(synthetic_higgs_df)
    assert len(X_tr) == 140
    assert len(X_val) == 30
    assert len(X_te) == 30
    assert len(y_tr) == 140

    # Ensure no column leakage into feature matrices
    assert TARGET_COLUMN not in X_tr.columns
    assert TARGET_COLUMN not in X_val.columns
    assert TARGET_COLUMN not in X_te.columns

    # Verify reproducibility with same seed
    (X_tr2, y_tr2), _, _ = split_dataset(
        synthetic_higgs_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )
    pd.testing.assert_frame_equal(X_tr, X_tr2)
    pd.testing.assert_series_equal(y_tr, y_tr2)


def test_load_sample_from_actual_dataset():
    """Verify loading real rows from the dataset zip archive."""
    sample_df = load_higgs_sample(n_rows=500)
    assert sample_df.shape == (500, 29)
    assert list(sample_df.columns) == COLUMN_NAMES
    assert sample_df[TARGET_COLUMN].isin([0.0, 1.0]).all()
    assert (sample_df.dtypes == np.float64).all()
