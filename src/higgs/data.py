"""Data loading for HIGGS: one-time CSV -> float32 array conversion, seeded stratified subsets, splits.

Source: the UCI download `higgs.zip`, which holds a single member `HIGGS.csv.gz`
(SHA-256 ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698). It is streamed
zip -> gzip -> CSV without extracting anything to disk.

Layout of the CSV (no header, 11,000,000 rows, 29 columns):
  col 0       label (1 = signal, 0 = background)
  cols 1-21   low-level kinematic measurements
  cols 22-28  high-level features derived by physicists from the low-level ones
The last 500,000 rows are the test set used by Baldi et al. (2014); we keep that convention.
"""
from __future__ import annotations

import gzip
import json
import os
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

N_ROWS = 11_000_000
N_COLS = 29

LOW_LEVEL = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
]
HIGH_LEVEL = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
FEATURE_NAMES = LOW_LEVEL + HIGH_LEVEL

# Column indices into the raw array (column 0 is the label).
FEATURE_SETS: dict[str, list[int]] = {
    "low": list(range(1, 22)),
    "high": list(range(22, 29)),
    "all": list(range(1, 29)),
}


# Engineered sets computed from the raw columns (see higgs/physics.py).
ENGINEERED_SETS = ["geo"]
ALL_FEATURE_SETS = list(FEATURE_SETS) + ENGINEERED_SETS


def feature_names(feature_set: str) -> list[str]:
    """Human-readable names for the columns of a feature set."""
    if feature_set == "geo":
        from higgs.physics import geometric_features
        return geometric_features(np.zeros((1, 28), dtype=np.float32), 1.0)[1]
    return [FEATURE_NAMES[i - 1] for i in FEATURE_SETS[feature_set]]


ZIP_MEMBER = "HIGGS.csv.gz"


@contextmanager
def open_csv_stream(raw_path: str | Path):
    """Binary stream of the CSV text from higgs.zip (zip -> gz member), a .csv.gz, or a plain .csv."""
    raw_path = Path(raw_path)
    if raw_path.suffix == ".zip":
        with zipfile.ZipFile(raw_path) as zf, zf.open(ZIP_MEMBER) as member, gzip.open(member) as f:
            yield f
    elif raw_path.suffix == ".gz":
        with gzip.open(raw_path) as f:
            yield f
    else:
        with open(raw_path, "rb") as f:
            yield f


def csv_to_array(raw_path: str | Path, array_path: str | Path, chunksize: int = 1_000_000) -> Path:
    """Stream the CSV out of higgs.zip in chunks into a float32 .npy file (memory-mappable). Idempotent."""
    array_path = Path(array_path)
    if array_path.exists():
        return array_path
    array_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = array_path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(N_ROWS, N_COLS))
    row = 0
    with open_csv_stream(raw_path) as stream:
        for chunk in pd.read_csv(stream, header=None, dtype=np.float32, chunksize=chunksize):
            values = chunk.to_numpy()
            out[row:row + len(values)] = values
            row += len(values)
            print(f"  converted {row:>11,} / {N_ROWS:,} rows", flush=True)
    out.flush()
    del out
    if row != N_ROWS:
        tmp.unlink()
        raise ValueError(f"expected {N_ROWS} rows, read {row}; the download is probably truncated")
    os.replace(tmp, array_path)
    return array_path


def open_array(array_path: str | Path) -> np.ndarray:
    """Read-only memory map of the full dataset; nothing is loaded until indexed."""
    return np.load(array_path, mmap_mode="r")


def make_split_indices(
    pool_labels: np.ndarray, subset_size: int, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified sample of `subset_size` rows from the pool, then a stratified train/val split.

    Returns sorted row indices into the pool. `subset_size >= len(pool_labels)` uses the whole pool.
    """
    idx = np.arange(len(pool_labels))
    if subset_size < len(pool_labels):
        idx, _ = train_test_split(idx, train_size=subset_size, stratify=pool_labels, random_state=seed)
    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, stratify=pool_labels[idx], random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


@dataclass
class Splits:
    """Raw (unscaled) arrays for one subset. X has all 28 features; select columns per feature set."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    manifest: dict

    def features(self, feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train/val/test feature matrices for one feature set (engineered ones fit on train only)."""
        if feature_set == "geo":
            from higgs.physics import fit_phi_scale, geometric_features
            scale = fit_phi_scale(self.X_train)
            return tuple(geometric_features(X, scale)[0] for X in (self.X_train, self.X_val, self.X_test))
        cols = [i - 1 for i in FEATURE_SETS[feature_set]]
        return self.X_train[:, cols], self.X_val[:, cols], self.X_test[:, cols]


def load_splits(cfg: dict) -> Splits:
    """Build (or reuse) the documented subset for cfg's seed and budget and load it into memory."""
    d = cfg["data"]
    data = open_array(d["array_path"])
    n_pool = len(data) - d["n_test"]
    subset_size = min(int(d["subset_size"]), n_pool)
    seed = int(cfg["seed"])

    splits_dir = Path(d["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    stem = splits_dir / f"subset_{subset_size}_seed{seed}"
    idx_path = stem.with_suffix(".npz")
    if idx_path.exists():
        saved = np.load(idx_path)
        train_idx, val_idx = saved["train"], saved["val"]
    else:
        pool_labels = np.asarray(data[:n_pool, 0])
        train_idx, val_idx = make_split_indices(pool_labels, subset_size, d["val_fraction"], seed)
        np.savez_compressed(idx_path, train=train_idx, val=val_idx)

    test = np.asarray(data[n_pool:])
    train = data[train_idx]
    val = data[val_idx]

    def counts(y: np.ndarray) -> dict:
        return {"n": int(len(y)), "signal": int(y.sum()), "signal_fraction": round(float(y.mean()), 5)}

    manifest = {
        "seed": seed,
        "subset_size": subset_size,
        "pool_rows": f"0..{n_pool - 1}",
        "test_rows": f"{n_pool}..{len(data) - 1}",
        "index_file": str(idx_path),
        "train": counts(train[:, 0]),
        "val": counts(val[:, 0]),
        "test": counts(test[:, 0]),
    }
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    # Labels are split off here, before any scaling ever sees the data.
    return Splits(
        train[:, 1:], train[:, 0].astype(np.int8),
        val[:, 1:], val[:, 0].astype(np.int8),
        test[:, 1:], test[:, 0].astype(np.int8),
        manifest,
    )
