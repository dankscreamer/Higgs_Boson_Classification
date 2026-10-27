"""
Unit tests for higgs.physics module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from higgs.data import (
    ALL_FEATURES,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
    load_higgs_sample,
)
from higgs.physics import (
    EXPECTED_DIMENSIONS,
    GEOMETRIC_FEATURE_NAMES,
    STANDARDIZED_PHI_MAX,
    compute_delta_phi,
    compute_delta_R,
    compute_geometric_features,
    get_feature_groups,
)


@pytest.fixture
def sample_higgs_df() -> pd.DataFrame:
    """Fixture providing a deterministic sample of real HIGGS events."""
    return load_higgs_sample(n_rows=500)


def test_angular_wraparound_hand_calculated():
    """Verify angular wraparound using exact hand-calculated edge cases in radians."""
    # Case 1: Angles close to +/- pi boundary
    # phi_a = +3.10 rad, phi_b = -3.10 rad
    # Arc distance across boundary = (pi - 3.10) + (pi - 3.10) = 2*(pi - 3.10)
    phi_a = 3.10
    phi_b = -3.10
    dphi = compute_delta_phi(phi_a, phi_b, use_radians=True, phi_scale=np.pi)
    expected = 2.0 * (np.pi - 3.10)
    assert np.isclose(dphi, expected, atol=1e-12)

    # Case 2: Orthogonal angles (0 and pi/2)
    dphi_ortho = compute_delta_phi(0.0, np.pi / 2.0, use_radians=True, phi_scale=np.pi)
    assert np.isclose(dphi_ortho, np.pi / 2.0, atol=1e-12)

    # Case 3: Exactly identical angles
    dphi_zero = compute_delta_phi(1.234, 1.234, use_radians=True, phi_scale=np.pi)
    assert np.isclose(dphi_zero, 0.0, atol=1e-12)

    # Case 4: Diametrically opposite angles (0 and pi)
    dphi_pi = compute_delta_phi(0.0, np.pi, use_radians=True, phi_scale=np.pi)
    assert np.isclose(dphi_pi, np.pi, atol=1e-12)


def test_standardized_angular_wraparound():
    """Verify wraparound in standardized coordinates with scaled period."""
    # Standardized boundary corresponds to +/- 1.7433
    phi_a = 1.70
    phi_b = -1.70
    # Expected scaled distance: 2 * (1.7433 - 1.70) = 0.0866
    dphi_std = compute_delta_phi(phi_a, phi_b, use_radians=False, phi_scale=STANDARDIZED_PHI_MAX)
    expected_std = 2.0 * (STANDARDIZED_PHI_MAX - 1.70)
    assert np.isclose(dphi_std, expected_std, atol=1e-10)


def test_rotation_invariance():
    """Verify invariance of delta_phi and delta_R under arbitrary global azimuthal rotation alpha."""
    rng = np.random.default_rng(123)
    phi_a = rng.uniform(-1.5, 1.5, size=100)
    phi_b = rng.uniform(-1.5, 1.5, size=100)
    eta_a = rng.normal(0, 1, size=100)
    eta_b = rng.normal(0, 1, size=100)

    dphi_orig = compute_delta_phi(phi_a, phi_b, use_radians=True)
    dR_orig = compute_delta_R(eta_a, eta_b, phi_a, phi_b, use_radians=True)

    # Rotate by arbitrary angle in standardized space
    # (equivalent to rotating physical azimuthal frame by alpha_rad)
    alpha_std = 0.85
    dphi_rot = compute_delta_phi(phi_a + alpha_std, phi_b + alpha_std, use_radians=True)
    dR_rot = compute_delta_R(eta_a, eta_b, phi_a + alpha_std, phi_b + alpha_std, use_radians=True)

    np.testing.assert_allclose(dphi_orig, dphi_rot, atol=1e-12)
    np.testing.assert_allclose(dR_orig, dR_rot, atol=1e-12)


def test_longitudinal_boost_invariance():
    """Verify delta_R invariance under longitudinal boost eta -> eta + beta."""
    eta_a = np.array([0.5, -1.2, 2.0])
    eta_b = np.array([-0.3, 0.8, -0.5])
    phi_a = np.array([0.2, -1.0, 1.5])
    phi_b = np.array([-0.4, 0.9, -1.2])

    dR_base = compute_delta_R(eta_a, eta_b, phi_a, phi_b)

    beta = 1.45  # longitudinal boost shift
    dR_boosted = compute_delta_R(eta_a + beta, eta_b + beta, phi_a, phi_b)

    np.testing.assert_allclose(dR_base, dR_boosted, atol=1e-12)


def test_geometric_features_dimensions_and_finiteness(sample_higgs_df):
    """Verify compute_geometric_features returns exactly 10 finite columns."""
    geo_df = compute_geometric_features(sample_higgs_df)

    assert geo_df.shape == (len(sample_higgs_df), 10)
    assert list(geo_df.columns) == GEOMETRIC_FEATURE_NAMES
    assert not geo_df.isna().any().any()
    assert np.isfinite(geo_df.values).all()


def test_no_target_leakage_in_geometric_features(sample_higgs_df):
    """Confirm target column is never used to construct geometric features."""
    df_no_target = sample_higgs_df.drop(columns=[TARGET_COLUMN])
    geo_df_1 = compute_geometric_features(sample_higgs_df)
    geo_df_2 = compute_geometric_features(df_no_target)

    pd.testing.assert_frame_equal(geo_df_1, geo_df_2)
    assert TARGET_COLUMN not in geo_df_1.columns


def test_get_feature_groups_dimensions(sample_higgs_df):
    """Verify all 5 feature groups match theoretical dimension specifications."""
    groups = get_feature_groups(sample_higgs_df)

    assert set(groups.keys()) == {"low", "high", "all", "geo", "geo_high"}

    assert groups["low"].shape == (len(sample_higgs_df), EXPECTED_DIMENSIONS["low"])
    assert groups["high"].shape == (len(sample_higgs_df), EXPECTED_DIMENSIONS["high"])
    assert groups["all"].shape == (len(sample_higgs_df), EXPECTED_DIMENSIONS["all"])
    assert groups["geo"].shape == (len(sample_higgs_df), EXPECTED_DIMENSIONS["geo"])
    assert groups["geo_high"].shape == (len(sample_higgs_df), EXPECTED_DIMENSIONS["geo_high"])

    assert groups["low"].shape[1] == 21
    assert groups["high"].shape[1] == 7
    assert groups["all"].shape[1] == 28
    assert groups["geo"].shape[1] == 10
    assert groups["geo_high"].shape[1] == 17

    # Ensure no target column in any feature group
    for name, grp in groups.items():
        assert TARGET_COLUMN not in grp.columns, f"Target leaked into feature group '{name}'"
