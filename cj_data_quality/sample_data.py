"""Utility module for loading and generating sample CJ data.

Provides convenience functions that other modules can use to quickly
obtain sample data for profiling, validation, and visualization demos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve the project layout so imports work whether this is called as a
# library (installed package) or from a script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_DATA_DIR = _PROJECT_ROOT / "data" / "sample"


def get_sample_data_path() -> Path:
    """Return the absolute path to the ``data/sample/`` directory.

    The directory is created if it does not already exist.
    """
    _SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _SAMPLE_DATA_DIR


def load_sample_data(filename: str = "corrections_data.csv") -> pd.DataFrame:
    """Load a CSV from the ``data/sample/`` directory.

    Parameters
    ----------
    filename:
        Name of the CSV file inside ``data/sample/``.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame with date columns parsed.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.  Callers can use
        :func:`generate_and_load` instead to auto-generate when missing.
    """
    path = get_sample_data_path() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Sample data file not found: {path}\n"
            "Run `python data/download_bjs_data.py` to generate it, "
            "or use `generate_and_load()` instead."
        )

    date_columns = [
        "admission_date",
        "release_date",
        "offense_date",
        "sentence_date",
        "reporting_date",
    ]
    # Only parse columns that actually exist in the file
    df = pd.read_csv(path)
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def generate_and_load(
    n_records: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data if the CSV does not exist, then load it.

    This is a convenience wrapper that combines generation and loading.
    If ``data/sample/corrections_data.csv`` already exists it is loaded
    directly; otherwise the generator is invoked first.

    Parameters
    ----------
    n_records:
        Number of records to generate (only used when the file is missing).
    seed:
        Random seed for reproducibility (only used when the file is missing).

    Returns
    -------
    pd.DataFrame
        The corrections dataset.
    """
    csv_path = get_sample_data_path() / "corrections_data.csv"

    if not csv_path.exists():
        # Lazy import to avoid circular deps and keep startup fast
        # when the file already exists.
        # We need to make sure the data package is importable.
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))

        from data.download_bjs_data import generate_synthetic_corrections_data

        df = generate_synthetic_corrections_data(
            n_records=n_records, seed=seed
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df

    return load_sample_data("corrections_data.csv")
