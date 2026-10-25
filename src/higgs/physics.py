"""Physics-motivated geometric features built from the 21 low-level columns (no invariant masses).

The collision is symmetric under rotation about the beam axis, so an absolute azimuthal angle phi
carries no information (EDA: single-feature AUC ~0.500). Only angles *between* objects can matter.
The `geo` feature set therefore drops absolute phi and adds:
  * |dphi| for every pair of {lepton, missing energy, jet1..jet4}   (15 features)
  * dR = sqrt(deta^2 + dphi^2) for every pair of {lepton, jet1..jet4} (10 features; MET has no eta)

UCI standardised every column. phi is uniform on [-pi, pi], so its scale is recovered from the
training data as pi / max|phi| and differences are wrapped to [-pi, pi]. eta's original scale
cannot be recovered exactly, but it is already ~unit scale, so dR is approximate along eta.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

# Column indices into the 28-feature matrix (label already removed).
OBJECTS: dict[str, dict[str, int]] = {
    "lepton": {"pt": 0, "eta": 1, "phi": 2},
    "met": {"pt": 3, "phi": 4},
    "jet1": {"pt": 5, "eta": 6, "phi": 7, "btag": 8},
    "jet2": {"pt": 9, "eta": 10, "phi": 11, "btag": 12},
    "jet3": {"pt": 13, "eta": 14, "phi": 15, "btag": 16},
    "jet4": {"pt": 17, "eta": 18, "phi": 19, "btag": 20},
}
PHI_COLS = [o["phi"] for o in OBJECTS.values()]
INVARIANT_COLS = [i for i in range(21) if i not in PHI_COLS]   # pT, eta, |MET|, b-tags


def fit_phi_scale(X_train: np.ndarray) -> float:
    """Factor converting standardised phi back to radians, estimated on training rows only."""
    return float(np.pi / np.abs(X_train[:, PHI_COLS]).max())


def _wrapped_abs_dphi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.abs((d + np.pi) % (2 * np.pi) - np.pi)


def geometric_features(X: np.ndarray, phi_scale: float) -> tuple[np.ndarray, list[str]]:
    """Rotation-invariant low-level features plus pairwise |dphi| and dR. Returns (matrix, names)."""
    from higgs.data import FEATURE_NAMES

    cols = [X[:, INVARIANT_COLS]]
    names = [FEATURE_NAMES[i] for i in INVARIANT_COLS]
    phi = {k: X[:, o["phi"]] * phi_scale for k, o in OBJECTS.items()}
    for a, b in combinations(OBJECTS, 2):
        cols.append(_wrapped_abs_dphi(phi[a], phi[b])[:, None])
        names.append(f"dphi_{a}_{b}")
    with_eta = [k for k, o in OBJECTS.items() if "eta" in o]
    for a, b in combinations(with_eta, 2):
        deta = X[:, OBJECTS[a]["eta"]] - X[:, OBJECTS[b]["eta"]]
        dphi = _wrapped_abs_dphi(phi[a], phi[b])
        cols.append(np.sqrt(deta**2 + dphi**2)[:, None])
        names.append(f"dR_{a}_{b}")
    return np.hstack(cols).astype(np.float32), names
