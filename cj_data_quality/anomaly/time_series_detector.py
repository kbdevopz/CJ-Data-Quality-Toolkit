"""Time-series anomaly detection: Z-score, IQR, rolling window, missing periods.

Provides four detection strategies for identifying anomalous values in
time-series criminal justice data (e.g., monthly population counts).
Each function returns a list of frozen ``AnomalyResult`` attrs instances.
"""

from datetime import date

import numpy as np
import pandas as pd

from cj_data_quality.anomaly._utils import to_date
from cj_data_quality.types import AnomalyResult, AnomalyType


def detect_zscore_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    threshold: float = 3.0,
    metric_name: str = "",
    state_code: str | None = None,
) -> list[AnomalyResult]:
    """Flag values whose absolute Z-score exceeds *threshold*.

    Z-score is computed over the entire series:
        z_i = (x_i - mean) / std

    Args:
        df: DataFrame containing a date column and a numeric value column.
        date_col: Name of the date/datetime column.
        value_col: Name of the numeric value column.
        threshold: Absolute Z-score cutoff (default 3.0).
        metric_name: Human-readable metric label for the result.
        state_code: Optional state code to attach to each result.

    Returns:
        A list of ``AnomalyResult`` instances, one per anomalous row.
    """
    if df.empty:
        return []
    for col in [date_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    values: pd.Series = df[value_col].astype(float)
    mean: float = float(values.mean())
    std: float = float(values.std(ddof=0))

    if std == 0:
        return []

    zscores: pd.Series = (values - mean) / std

    results: list[AnomalyResult] = []
    for idx in df.index:
        z: float = float(zscores.loc[idx])
        if abs(z) > threshold:
            ts: date = to_date(df.loc[idx, date_col])
            results.append(
                AnomalyResult(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.ZSCORE,
                    timestamp=ts,
                    observed_value=float(values.loc[idx]),
                    expected_value=mean,
                    deviation=abs(z),
                    threshold=threshold,
                    state_code=state_code,
                )
            )

    return results


def detect_iqr_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    multiplier: float = 1.5,
    metric_name: str = "",
    state_code: str | None = None,
) -> list[AnomalyResult]:
    """Flag values outside the IQR fence [Q1 - k*IQR, Q3 + k*IQR].

    Args:
        df: DataFrame containing a date column and a numeric value column.
        date_col: Name of the date/datetime column.
        value_col: Name of the numeric value column.
        multiplier: IQR multiplier *k* (default 1.5).
        metric_name: Human-readable metric label for the result.
        state_code: Optional state code to attach to each result.

    Returns:
        A list of ``AnomalyResult`` instances, one per anomalous row.
    """
    if df.empty:
        return []
    for col in [date_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    values: pd.Series = df[value_col].astype(float)
    q1: float = float(values.quantile(0.25))
    q3: float = float(values.quantile(0.75))
    iqr: float = q3 - q1
    lower_fence: float = q1 - multiplier * iqr
    upper_fence: float = q3 + multiplier * iqr
    median: float = float(values.median())

    results: list[AnomalyResult] = []
    for idx in df.index:
        val: float = float(values.loc[idx])
        if val < lower_fence or val > upper_fence:
            ts: date = to_date(df.loc[idx, date_col])
            deviation: float = (
                (val - upper_fence) if val > upper_fence else (lower_fence - val)
            )
            results.append(
                AnomalyResult(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.IQR,
                    timestamp=ts,
                    observed_value=val,
                    expected_value=median,
                    deviation=abs(deviation),
                    threshold=multiplier,
                    state_code=state_code,
                )
            )

    return results


def detect_rolling_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int = 12,
    num_std: float = 2.0,
    metric_name: str = "",
    state_code: str | None = None,
) -> list[AnomalyResult]:
    """Flag values outside rolling mean +/- num_std * rolling_std.

    A rolling window of *window* periods is used to compute a local mean and
    standard deviation.  Points that fall outside the band are flagged.

    Args:
        df: DataFrame containing a date column and a numeric value column.
            Must be sorted (or will be sorted) by *date_col*.
        date_col: Name of the date/datetime column.
        value_col: Name of the numeric value column.
        window: Rolling window size in periods (default 12).
        num_std: Number of standard deviations for the band (default 2.0).
        metric_name: Human-readable metric label for the result.
        state_code: Optional state code to attach to each result.

    Returns:
        A list of ``AnomalyResult`` instances, one per anomalous row.
    """
    if df.empty or len(df) < window:
        return []
    for col in [date_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    sorted_df: pd.DataFrame = df.sort_values(date_col).reset_index(drop=True)
    values: pd.Series = sorted_df[value_col].astype(float)

    rolling_mean: pd.Series = values.rolling(window=window, min_periods=window).mean()
    rolling_std: pd.Series = values.rolling(window=window, min_periods=window).std(ddof=0)

    results: list[AnomalyResult] = []
    for idx in sorted_df.index:
        rm: float = rolling_mean.iloc[idx]
        rs: float = rolling_std.iloc[idx]
        if np.isnan(rm) or np.isnan(rs) or rs == 0:
            continue

        val: float = float(values.iloc[idx])
        upper: float = rm + num_std * rs
        lower: float = rm - num_std * rs

        if val > upper or val < lower:
            ts: date = to_date(sorted_df.loc[idx, date_col])
            deviation: float = abs(val - rm) / rs
            results.append(
                AnomalyResult(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.ROLLING_WINDOW,
                    timestamp=ts,
                    observed_value=val,
                    expected_value=float(rm),
                    deviation=deviation,
                    threshold=num_std,
                    state_code=state_code,
                )
            )

    return results


def detect_missing_periods(
    df: pd.DataFrame,
    date_col: str,
    expected_freq: str = "MS",
    metric_name: str = "",
    state_code: str | None = None,
) -> list[AnomalyResult]:
    """Identify gaps in the expected reporting cadence.

    Builds the full expected date range from the minimum to maximum date in
    the data using *expected_freq*, then flags any dates that are absent.

    Args:
        df: DataFrame containing a date column.
        date_col: Name of the date/datetime column.
        expected_freq: pandas frequency string (default ``"MS"`` = month start).
        metric_name: Human-readable metric label for the result.
        state_code: Optional state code to attach to each result.

    Returns:
        A list of ``AnomalyResult`` instances, one per missing period.
    """
    if df.empty:
        return []
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    dates: pd.Series = pd.to_datetime(df[date_col])
    min_date: pd.Timestamp = dates.min()
    max_date: pd.Timestamp = dates.max()

    expected_dates: pd.DatetimeIndex = pd.date_range(
        start=min_date, end=max_date, freq=expected_freq
    )
    actual_dates: set[pd.Timestamp] = set(dates.dt.normalize())

    results: list[AnomalyResult] = []
    for expected_ts in expected_dates:
        if expected_ts not in actual_dates:
            results.append(
                AnomalyResult(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.MISSING_PERIOD,
                    timestamp=expected_ts.date(),
                    observed_value=0.0,
                    expected_value=1.0,
                    deviation=1.0,
                    threshold=0.0,
                    state_code=state_code,
                )
            )

    return results


