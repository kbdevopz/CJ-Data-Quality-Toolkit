"""Spike / drop detection and per-state population anomaly detection.

Provides period-over-period percent-change analysis and per-state Z-score
anomaly detection on population columns.  Each function returns a list of
frozen ``AnomalyResult`` attrs instances.
"""

from datetime import date

import numpy as np
import pandas as pd

from cj_data_quality.anomaly._utils import to_date
from cj_data_quality.types import AnomalyResult, AnomalyType


def detect_spikes(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    pct_change_threshold: float = 0.5,
    metric_name: str = "",
    state_code: str | None = None,
) -> list[AnomalyResult]:
    """Flag rows where period-over-period percent change exceeds *threshold*.

    Positive changes that exceed the threshold are classified as
    ``AnomalyType.SUDDEN_SPIKE``; negative changes that exceed the threshold
    (in absolute value) are classified as ``AnomalyType.SUDDEN_DROP``.

    Args:
        df: DataFrame containing a date column and a numeric value column.
        date_col: Name of the date/datetime column.
        value_col: Name of the numeric value column.
        pct_change_threshold: Absolute percent-change cutoff (default 0.5,
            meaning 50%).
        metric_name: Human-readable metric label for the result.
        state_code: Optional state code to attach to each result.

    Returns:
        A list of ``AnomalyResult`` instances, one per flagged row.
    """
    if df.empty or len(df) < 2:
        return []
    for col in [date_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    sorted_df: pd.DataFrame = df.sort_values(date_col).reset_index(drop=True)
    values: pd.Series = sorted_df[value_col].astype(float)
    pct_changes: pd.Series = values.pct_change()

    results: list[AnomalyResult] = []
    for idx in range(1, len(sorted_df)):
        pct: float = float(pct_changes.iloc[idx])
        if np.isnan(pct):
            continue

        if abs(pct) > pct_change_threshold:
            anomaly_type: AnomalyType = (
                AnomalyType.SUDDEN_SPIKE if pct > 0 else AnomalyType.SUDDEN_DROP
            )
            ts: date = to_date(sorted_df.loc[idx, date_col])
            previous_value: float = float(values.iloc[idx - 1])
            results.append(
                AnomalyResult(
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    timestamp=ts,
                    observed_value=float(values.iloc[idx]),
                    expected_value=previous_value,
                    deviation=abs(pct),
                    threshold=pct_change_threshold,
                    state_code=state_code,
                )
            )

    return results


def detect_population_anomalies(
    df: pd.DataFrame,
    date_col: str,
    population_col: str,
    state_col: str,
    threshold: float = 3.0,
) -> list[AnomalyResult]:
    """Per-state Z-score anomaly detection on a population column.

    For each state, the Z-score is computed over that state's time series.
    Rows with |z| > *threshold* are flagged as ``AnomalyType.ZSCORE``.

    Args:
        df: DataFrame containing date, population, and state columns.
        date_col: Name of the date/datetime column.
        population_col: Name of the numeric population column.
        state_col: Name of the state code column.
        threshold: Absolute Z-score cutoff (default 3.0).

    Returns:
        A list of ``AnomalyResult`` instances across all states.
    """
    if df.empty:
        return []
    for col in [date_col, population_col, state_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    results: list[AnomalyResult] = []

    for state, group in df.groupby(state_col):
        values: pd.Series = group[population_col].astype(float)
        mean: float = float(values.mean())
        std: float = float(values.std(ddof=0))

        if std == 0:
            continue

        zscores: pd.Series = (values - mean) / std

        for idx in group.index:
            z: float = float(zscores.loc[idx])
            if abs(z) > threshold:
                ts: date = to_date(df.loc[idx, date_col])
                results.append(
                    AnomalyResult(
                        metric_name=population_col,
                        anomaly_type=AnomalyType.ZSCORE,
                        timestamp=ts,
                        observed_value=float(values.loc[idx]),
                        expected_value=mean,
                        deviation=abs(z),
                        threshold=threshold,
                        state_code=str(state),
                    )
                )

    return results


# ---------------------------------------------------------------------------
