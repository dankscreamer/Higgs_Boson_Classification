"""
Physics-derived geometric transformations for the UCI HIGGS dataset.

This module implements Lorentz-boost invariant and rotation-invariant
geometric features (angular distances delta_phi and spatial cone separations delta_R)
derived strictly from low-level kinematic detector observables.

Mathematical & Scaling Context:
In high-energy physics, azimuthal angles phi are defined in physical radians [-pi, pi].
In the UCI HIGGS dataset (Baldi et al., Nature Communications 2014), input features
were globally standardized to mean 0 and standard deviation 1.
Because of this pre-scaling, the empirical range of phi features is [-1.7433, +1.7433]
rather than [-pi, pi]. Applying standard modulo 2*pi arithmetic directly to these
scaled values without unscaling would fail to enforce correct circular periodic wrap-around.
This module explicitly accounts for the standardization scale factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from higgs.data import (
    ALL_FEATURES,
    HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURES,
    TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# Scaling Constants & Feature Definitions
# ---------------------------------------------------------------------------

# Empirical boundary of standardized phi features in the UCI HIGGS dataset
# corresponding to the physical boundary of +/- pi radians.
STANDARDIZED_PHI_MAX: float = 1.7433

# Scale factor to transform standardized phi coordinates to approximate physical radians:
# phi_rad = phi_std * (pi / STANDARDIZED_PHI_MAX)
PHI_UNSCALING_FACTOR: float = float(np.pi / STANDARDIZED_PHI_MAX)

# Exactly 10 physically motivated geometric features:
# - 6 pairwise spatial cone separations delta_R between reconstructed objects
# - 3 transverse angular separations delta_phi with missing transverse energy (MET)
# - 1 dijet pseudorapidity gap delta_eta
GEOMETRIC_FEATURE_NAMES: list[str] = [
    "delta_R_lep_jet1",
    "delta_R_lep_jet2",
    "delta_R_jet1_jet2",
    "delta_R_jet1_jet3",
    "delta_R_jet2_jet3",
    "delta_R_jet3_jet4",
    "delta_phi_lep_met",
    "delta_phi_jet1_met",
    "delta_phi_jet2_met",
    "delta_eta_jet1_jet2",
]

# Expected dimension specifications
EXPECTED_DIMENSIONS: dict[str, int] = {
    "low": 21,
    "high": 7,
    "all": 28,
    "geo": 10,
    "geo_high": 17,
}


# ---------------------------------------------------------------------------
# Core Geometric Functions
# ---------------------------------------------------------------------------

def compute_delta_phi(
    phi_a: np.ndarray | pd.Series | float,
    phi_b: np.ndarray | pd.Series | float,
    use_radians: bool = True,
    phi_scale: float = STANDARDIZED_PHI_MAX,
) -> np.ndarray | pd.Series | float:
    """
    Computes the shortest angular difference on the circle between two azimuthal angles.

    For angles in radians [-pi, pi]:
        delta_phi = |((phi_a - phi_b + pi) % (2*pi)) - pi|  in [0, pi]

    Because the UCI dataset features were standardized, if use_radians=True,
    the standardized values are first unscaled to approximate radians using
    phi_scale before applying standard 2*pi periodicity.
    If use_radians=False, periodic arithmetic is applied directly using the
    scaled period T = 2 * phi_scale.

    Parameters
    ----------
    phi_a : array-like or float
        Azimuthal angle(s) of first particle/object.
    phi_b : array-like or float
        Azimuthal angle(s) of second particle/object.
    use_radians : bool
        Whether to unscale to physical radians before applying modulo 2*pi (default True).
    phi_scale : float
        Empirical boundary of standardized phi corresponding to pi radians (default 1.7433).

    Returns
    -------
    array-like or float
        Shortest angular distance on the circle.
    """
    if use_radians:
        scale_factor = np.pi / phi_scale
        pa = phi_a * scale_factor
        pb = phi_b * scale_factor
        period = 2.0 * np.pi
        half_period = np.pi
    else:
        pa = phi_a
        pb = phi_b
        period = 2.0 * phi_scale
        half_period = phi_scale

    diff = (pa - pb + half_period) % period - half_period
    return np.abs(diff)


def compute_delta_R(
    eta_a: np.ndarray | pd.Series | float,
    eta_b: np.ndarray | pd.Series | float,
    phi_a: np.ndarray | pd.Series | float,
    phi_b: np.ndarray | pd.Series | float,
    use_radians: bool = True,
    phi_scale: float = STANDARDIZED_PHI_MAX,
) -> np.ndarray | pd.Series | float:
    """
    Computes the spatial separation cone delta_R in (eta, phi) space.

        delta_R = sqrt(delta_eta^2 + delta_phi^2)

    Physics Motivation:
    In collider detectors, delta_eta is invariant under longitudinal Lorentz boosts
    along the beam axis for relativistic particles, and delta_phi is invariant under
    rotations around the beam axis. Hence, delta_R is a standard Lorentz-invariant
    metric used by experimental collaborations (ATLAS and CMS) to identify particle
    isolation and jet decay cones.

    Parameters
    ----------
    eta_a, eta_b : array-like or float
        Pseudorapidities of the two particles.
    phi_a, phi_b : array-like or float
        Azimuthal angles of the two particles.
    use_radians : bool
        Whether to compute delta_phi in physical radians (default True).
    phi_scale : float
        Standardized boundary corresponding to pi (default 1.7433).

    Returns
    -------
    array-like or float
        Spatial separation delta_R.
    """
    deta = eta_a - eta_b
    dphi = compute_delta_phi(phi_a, phi_b, use_radians=use_radians, phi_scale=phi_scale)
    return np.sqrt(deta**2 + dphi**2)


# ---------------------------------------------------------------------------
# Feature Engineering Transformation
# ---------------------------------------------------------------------------

def compute_geometric_features(
    df: pd.DataFrame, use_radians: bool = True
) -> pd.DataFrame:
    """
    Constructs the 10 physically justified geometric features derived strictly
    from the 21 low-level kinematic detector observables.

    Strict Leakage Prevention Guarantee:
    This function never accesses or references the target column `class_label`.
    It operates solely on the verified detector kinematic coordinates.

    Engineered Features:
    1. delta_R_lep_jet1   : Spatial cone between lepton and leading jet
    2. delta_R_lep_jet2   : Spatial cone between lepton and sub-leading jet
    3. delta_R_jet1_jet2  : Dijet separation (reconstructs hadronic W -> jj)
    4. delta_R_jet1_jet3  : Jet 1 to Jet 3 spatial separation
    5. delta_R_jet2_jet3  : Jet 2 to Jet 3 spatial separation
    6. delta_R_jet3_jet4  : Sub-leading dijet separation
    7. delta_phi_lep_met  : Transverse angle between lepton and MET (W -> l nu)
    8. delta_phi_jet1_met : Transverse angle between leading jet and MET
    9. delta_phi_jet2_met : Transverse angle between sub-leading jet and MET
    10. delta_eta_jet1_jet2: Dijet pseudorapidity gap |eta_j1 - eta_j2|

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 21 low-level kinematic features.
    use_radians : bool
        Whether to unscale standardized phi to physical radians for periodic wrap-around.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (N, 10) with exact columns GEOMETRIC_FEATURE_NAMES.

    Raises
    ------
    KeyError
        If any required low-level kinematic column is missing.
    ValueError
        If any computed feature contains NaN or infinite values.
    """
    required_cols = [
        "lepton_eta",
        "lepton_phi",
        "missing_energy_phi",
        "jet_1_eta",
        "jet_1_phi",
        "jet_2_eta",
        "jet_2_phi",
        "jet_3_eta",
        "jet_3_phi",
        "jet_4_eta",
        "jet_4_phi",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required low-level columns for geometric engineering: {missing}")

    geo_data: dict[str, pd.Series | np.ndarray] = {}

    # 1-6: Pairwise spatial cone separations delta_R
    geo_data["delta_R_lep_jet1"] = compute_delta_R(
        df["lepton_eta"], df["jet_1_eta"], df["lepton_phi"], df["jet_1_phi"], use_radians
    )
    geo_data["delta_R_lep_jet2"] = compute_delta_R(
        df["lepton_eta"], df["jet_2_eta"], df["lepton_phi"], df["jet_2_phi"], use_radians
    )
    geo_data["delta_R_jet1_jet2"] = compute_delta_R(
        df["jet_1_eta"], df["jet_2_eta"], df["jet_1_phi"], df["jet_2_phi"], use_radians
    )
    geo_data["delta_R_jet1_jet3"] = compute_delta_R(
        df["jet_1_eta"], df["jet_3_eta"], df["jet_1_phi"], df["jet_3_phi"], use_radians
    )
    geo_data["delta_R_jet2_jet3"] = compute_delta_R(
        df["jet_2_eta"], df["jet_3_eta"], df["jet_2_phi"], df["jet_3_phi"], use_radians
    )
    geo_data["delta_R_jet3_jet4"] = compute_delta_R(
        df["jet_3_eta"], df["jet_4_eta"], df["jet_3_phi"], df["jet_4_phi"], use_radians
    )

    # 7-9: Transverse plane opening angles with MET (MET has no pseudorapidity eta)
    geo_data["delta_phi_lep_met"] = compute_delta_phi(
        df["lepton_phi"], df["missing_energy_phi"], use_radians
    )
    geo_data["delta_phi_jet1_met"] = compute_delta_phi(
        df["jet_1_phi"], df["missing_energy_phi"], use_radians
    )
    geo_data["delta_phi_jet2_met"] = compute_delta_phi(
        df["jet_2_phi"], df["missing_energy_phi"], use_radians
    )

    # 10: Pseudorapidity gap between leading jets
    geo_data["delta_eta_jet1_jet2"] = np.abs(df["jet_1_eta"] - df["jet_2_eta"])

    geo_df = pd.DataFrame(geo_data, index=df.index)

    # Validate output finite values
    nan_count = int(geo_df.isna().sum().sum())
    inf_count = int(np.isinf(geo_df.values).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Geometric feature computation produced non-finite values (NaN: {nan_count}, Inf: {inf_count})"
        )

    return geo_df


# ---------------------------------------------------------------------------
# Feature Group Assembly & Dimension Validation
# ---------------------------------------------------------------------------

def get_feature_groups(
    df: pd.DataFrame, use_radians: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Constructs and returns the five standardized feature representations,
    strictly validating actual output dimensions against theoretical specifications.

    Feature Groups:
    1. `low`     : 21 low-level kinematic detector observables.
    2. `high`    : 7 physicist-derived invariant mass reconstructions.
    3. `all`     : 28 original features (`low` + `high`).
    4. `geo`     : 10 justified geometric features derived from low-level observables.
    5. `geo_high`: 17 features combining geometric features with the 7 high-level mass features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw HIGGS columns.
    use_radians : bool
        Whether to unscale standardized phi to physical radians.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping group names ('low', 'high', 'all', 'geo', 'geo_high')
        to their respective feature DataFrames.

    Raises
    ------
    ValueError
        If any feature group fails the strict dimension validation check.
    """
    # 1. Low-level (21)
    df_low = df[LOW_LEVEL_FEATURES].copy()

    # 2. High-level (7)
    df_high = df[HIGH_LEVEL_FEATURES].copy()

    # 3. All original (28)
    df_all = df[ALL_FEATURES].copy()

    # 4. Geometric (10)
    df_geo = compute_geometric_features(df, use_radians=use_radians)

    # 5. Geometric + High-level (10 + 7 = 17)
    df_geo_high = pd.concat([df_geo, df_high], axis=1)

    feature_groups = {
        "low": df_low,
        "high": df_high,
        "all": df_all,
        "geo": df_geo,
        "geo_high": df_geo_high,
    }

    # Strict dimension validation
    for name, grp in feature_groups.items():
        expected_dim = EXPECTED_DIMENSIONS[name]
        actual_dim = grp.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Feature group '{name}' dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )
        # Verify no target leakage
        if TARGET_COLUMN in grp.columns:
            raise ValueError(f"Target leakage detected in feature group '{name}'!")

    return feature_groups
