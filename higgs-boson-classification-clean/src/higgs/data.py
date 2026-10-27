"""
Data loading, schema validation, and dataset utilities for the UCI HIGGS dataset.

Dataset Reference:
Baldi, P., Sadowski, P., & Whiteson, D. (2014).
Searching for Exotic Particles in High-Energy Physics with Deep Learning.
Nature Communications, 5, 4308.
"""

from __future__ import annotations

import gzip
import os
import zipfile
from pathlib import Path
from typing import Generator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Official UCI HIGGS Schema Definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN: str = "class_label"

LOW_LEVEL_FEATURES: list[str] = [
    "lepton_pT",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet_1_pt",
    "jet_1_eta",
    "jet_1_phi",
    "jet_1_b-tag",
    "jet_2_pt",
    "jet_2_eta",
    "jet_2_phi",
    "jet_2_b-tag",
    "jet_3_pt",
    "jet_3_eta",
    "jet_3_phi",
    "jet_3_b-tag",
    "jet_4_pt",
    "jet_4_eta",
    "jet_4_phi",
    "jet_4_b-tag",
]

HIGH_LEVEL_FEATURES: list[str] = [
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]

ALL_FEATURES: list[str] = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES

COLUMN_NAMES: list[str] = [TARGET_COLUMN] + ALL_FEATURES

assert len(LOW_LEVEL_FEATURES) == 21, "Expected exactly 21 low-level features"
assert len(HIGH_LEVEL_FEATURES) == 7, "Expected exactly 7 high-level features"
assert len(COLUMN_NAMES) == 29, "Expected exactly 29 columns (1 target + 28 features)"


# ---------------------------------------------------------------------------
# Dataset Path Resolution & Stream Handling
# ---------------------------------------------------------------------------

def resolve_dataset_path(data_path: str | Path | None = None) -> Path:
    """
    Locates the HIGGS dataset file across candidate locations.

    Candidate search locations:
    1. Explicitly provided `data_path`
    2. Environment variable `HIGGS_DATA_PATH`
    3. `data/HIGGS.csv.gz`, `data/HIGGS.csv`, `data/higgs.zip`, `data/higgs (1).zip`
    4. Parent directory fallbacks (`../higgs (1).zip`, `../HIGGS.csv.gz`)

    Parameters
    ----------
    data_path : str | Path | None
        Explicit file path if available.

    Returns
    -------
    Path
        Resolved existing path to the dataset file.

    Raises
    ------
    FileNotFoundError
        If no valid dataset file can be found.
    """
    if data_path is not None:
        p = Path(data_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified dataset path does not exist: {p}")

    env_path = os.getenv("HIGGS_DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    # Candidate relative to CWD and package/project locations
    pkg_dir = Path(__file__).resolve().parent
    project_dir = pkg_dir.parent.parent  # higgs-boson-classification-clean
    workspace_dir = project_dir.parent   # higgs_boson

    candidate_dirs = [
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd().parent,
        Path.cwd().parent / "data",
        project_dir,
        project_dir / "data",
        workspace_dir,
        workspace_dir / "data",
    ]

    target_filenames = [
        "HIGGS.csv.gz",
        "HIGGS.csv",
        "higgs (1).zip",
        "higgs.zip",
    ]

    for cdir in candidate_dirs:
        for fname in target_filenames:
            candidate = (cdir / fname).resolve()
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not find the HIGGS dataset file. Please place 'HIGGS.csv.gz', "
        "'HIGGS.csv', or 'higgs (1).zip' in the 'data/' directory, or set "
        "the HIGGS_DATA_PATH environment variable."
    )


def open_dataset_stream(file_path: Path):
    """
    Context manager returning a readable text stream for the dataset,
    handling uncompressed CSV, gzipped CSV, or ZIP files containing gzip/CSV.

    Parameters
    ----------
    file_path : Path
        Path to .csv, .csv.gz, or .zip archive.

    Yields
    ------
    TextIO
        A readable file-like stream of text lines.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".zip":
        z = zipfile.ZipFile(file_path, mode="r")
        # Identify the data entry inside the zip
        names = z.namelist()
        gz_members = [n for n in names if n.endswith(".csv.gz")]
        csv_members = [n for n in names if n.endswith(".csv")]

        if gz_members:
            member = gz_members[0]
            raw_stream = z.open(member, "r")
            gz_stream = gzip.GzipFile(fileobj=raw_stream, mode="rb")
            import io
            text_stream = io.TextIOWrapper(gz_stream, encoding="utf-8")
            return text_stream
        elif csv_members:
            member = csv_members[0]
            raw_stream = z.open(member, "r")
            import io
            text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8")
            return text_stream
        else:
            raise ValueError(f"Zip archive {file_path} contains neither .csv nor .csv.gz: {names}")

    elif suffix == ".gz":
        return gzip.open(file_path, mode="rt", encoding="utf-8")
    elif suffix == ".csv":
        return open(file_path, mode="r", encoding="utf-8")
    else:
        # Fallback to standard open
        return open(file_path, mode="r", encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset Loading Functions
# ---------------------------------------------------------------------------

def load_higgs_sample(
    n_rows: int = 100_000,
    data_path: str | Path | None = None,
    use_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Loads a deterministic sample of rows from the HIGGS dataset.

    Because the raw CSV does not have a header, column names are assigned
    according to the official UCI HIGGS schema.

    Parameters
    ----------
    n_rows : int
        Number of rows to load from the start of the dataset.
    data_path : str | Path | None
        Optional explicit path to the dataset.
    use_cols : Sequence[str] | None
        Optional subset of column names to load.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with verified column names and float64 dtypes.
    """
    path = resolve_dataset_path(data_path)
    stream = open_dataset_stream(path)

    try:
        df = pd.read_csv(
            stream,
            header=None,
            names=COLUMN_NAMES,
            usecols=use_cols,
            nrows=n_rows,
            dtype=np.float64,
        )
    finally:
        stream.close()

    return df


def stream_higgs_chunks(
    chunk_size: int = 250_000,
    max_rows: int | None = None,
    data_path: str | Path | None = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yields chunks of the HIGGS dataset for memory-efficient full-dataset
    or large-scale streaming inspection.

    Parameters
    ----------
    chunk_size : int
        Number of rows per yielded chunk.
    max_rows : int | None
        Optional upper limit on total rows to stream across all chunks.
    data_path : str | Path | None
        Optional explicit dataset path.

    Yields
    ------
    pd.DataFrame
        Chunk DataFrame with official schema applied.
    """
    path = resolve_dataset_path(data_path)
    stream = open_dataset_stream(path)

    try:
        reader = pd.read_csv(
            stream,
            header=None,
            names=COLUMN_NAMES,
            chunksize=chunk_size,
            dtype=np.float64,
        )
        rows_read = 0
        for chunk in reader:
            if max_rows is not None and rows_read + len(chunk) > max_rows:
                remaining = max_rows - rows_read
                if remaining > 0:
                    yield chunk.iloc[:remaining]
                break
            yield chunk
            rows_read += len(chunk)
    finally:
        stream.close()


# ---------------------------------------------------------------------------
# Stage 1 Dataset Validation Functions
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> dict[str, object]:
    """
    Validates that a DataFrame conforms to the expected 29-column HIGGS schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    dict[str, object]
        Validation diagnostic report containing check statuses and details.

    Raises
    ------
    ValueError
        If column count or column names do not match the expected schema.
    """
    report: dict[str, object] = {}

    expected_cols = 29
    actual_cols = df.shape[1]
    report["column_count_matches"] = actual_cols == expected_cols
    report["actual_columns"] = actual_cols
    report["expected_columns"] = expected_cols

    if actual_cols != expected_cols:
        raise ValueError(
            f"Schema mismatch: expected {expected_cols} columns, found {actual_cols}"
        )

    expected_names = COLUMN_NAMES
    actual_names = list(df.columns)
    report["column_names_match"] = actual_names == expected_names

    if actual_names != expected_names:
        raise ValueError(
            f"Column names do not match expected schema.\n"
            f"Expected: {expected_names}\nFound: {actual_names}"
        )

    # Check numeric types
    non_numeric_cols = [
        col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)
    ]
    report["all_numeric"] = len(non_numeric_cols) == 0
    report["non_numeric_columns"] = non_numeric_cols

    if non_numeric_cols:
        raise ValueError(f"Found non-numeric columns in dataset: {non_numeric_cols}")

    return report


def validate_target_labels(df: pd.DataFrame) -> dict[str, object]:
    """
    Validates that the target label contains strictly binary values (0.0 and 1.0).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing `TARGET_COLUMN`.

    Returns
    -------
    dict[str, object]
        Target label diagnostic report.

    Raises
    ------
    ValueError
        If invalid target values are discovered.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    unique_vals = set(df[TARGET_COLUMN].dropna().unique())
    expected_vals = {0.0, 1.0}

    is_valid = unique_vals.issubset(expected_vals)
    nan_count = int(df[TARGET_COLUMN].isna().sum())

    report = {
        "unique_values": sorted(list(unique_vals)),
        "is_binary": is_valid,
        "missing_target_count": nan_count,
    }

    if not is_valid:
        raise ValueError(
            f"Target contains invalid non-binary values: {unique_vals}. Expected {expected_vals}"
        )
    if nan_count > 0:
        raise ValueError(f"Target column contains {nan_count} missing values.")

    return report


def check_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes class counts, relative frequencies, and signal-to-background ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing `TARGET_COLUMN`.

    Returns
    -------
    pd.DataFrame
        Formatted summary table showing counts and percentages.
    """
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame(
        {
            "Class": ["Background (0.0)", "Signal (1.0)"],
            "Count": [int(counts.get(0.0, 0)), int(counts.get(1.0, 0))],
            "Percentage (%)": [
                round(float(percentages.get(0.0, 0.0)), 2),
                round(float(percentages.get(1.0, 0.0)), 2),
            ],
        }
    )
    return summary


def check_missing_and_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans every column for missing (NaN/None) and infinite (+inf/-inf) values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    pd.DataFrame
        Summary table detailing any occurrences of NaN or Inf values.
    """
    nan_counts = df.isna().sum()
    pos_inf = np.isposinf(df.select_dtypes(include=[np.number])).sum()
    neg_inf = np.isneginf(df.select_dtypes(include=[np.number])).sum()

    report = pd.DataFrame(
        {
            "Missing (NaN)": nan_counts,
            "+Infinity": pos_inf,
            "-Infinity": neg_inf,
        }
    )
    report["Total Anomalies"] = (
        report["Missing (NaN)"] + report["+Infinity"] + report["-Infinity"]
    )
    return report


def check_constant_features(
    df: pd.DataFrame, variance_threshold: float = 1e-5
) -> pd.DataFrame:
    """
    Checks for constant (variance == 0) and near-constant features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    variance_threshold : float
        Threshold below which a feature is flagged as near-constant.

    Returns
    -------
    pd.DataFrame
        Summary containing variance, standard deviation, min, max,
        and unique count for each feature.
    """
    features = [c for c in df.columns if c != TARGET_COLUMN]
    stats = []

    for f in features:
        series = df[f]
        var = float(series.var())
        std = float(series.std())
        vmin = float(series.min())
        vmax = float(series.max())
        n_unique = int(series.nunique())
        is_constant = bool(var == 0.0)
        is_near_constant = bool(var < variance_threshold)

        stats.append(
            {
                "Feature": f,
                "Variance": var,
                "Std Dev": std,
                "Min": vmin,
                "Max": vmax,
                "N Unique": n_unique,
                "Is Constant": is_constant,
                "Is Near-Constant": is_near_constant,
            }
        )

    return pd.DataFrame(stats)


def check_duplicates(
    df: pd.DataFrame, subset: Sequence[str] | None = None
) -> dict[str, object]:
    """
    Counts duplicate rows in the provided DataFrame or subset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    subset : Sequence[str] | None
        Optional columns to consider when identifying duplicates.

    Returns
    -------
    dict[str, object]
        Summary containing row count, duplicate count, and duplicate rate.
    """
    dup_count = int(df.duplicated(subset=subset).sum())
    total_rows = len(df)
    dup_percentage = (dup_count / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "total_rows_inspected": total_rows,
        "duplicate_rows": dup_count,
        "duplicate_percentage": round(dup_percentage, 4),
    }


# ---------------------------------------------------------------------------
# Reproducible Data Splitting
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
]:
    """
    Splits the dataset into train, validation, and test sets reproducibly.

    Maintains strict separation:
    - Train set: used for fitting models and preprocessing transforms.
    - Validation set: used for hyperparameter tuning and model selection.
    - Test set: held out completely for final evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Full or sampled dataset.
    train_size : float
        Proportion for training (default 0.70).
    val_size : float
        Proportion for validation (default 0.15).
    test_size : float
        Proportion for test (default 0.15).
    random_state : int
        Seed for reproducibility.
    stratify : bool
        Whether to preserve class balance across all splits.

    Returns
    -------
    tuple
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split proportions must sum to 1.0 (got {train_size} + {val_size} + {test_size} = {total})"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    stratify_target = y if stratify else None

    # Step 1: Split into train and temp (val + test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Step 2: Split temp into val and test
    relative_test_size = test_size / temp_size
    stratify_temp = y_temp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
