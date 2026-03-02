"""Column-level profiling: compute statistics for individual columns.

Produces ColumnProfile instances with null rates, cardinality, top-k values,
and optional NumericStats / TemporalStats depending on the column type.
"""

from datetime import date

import pandas as pd

from cj_data_quality.profiling.type_inference import infer_column_type
from cj_data_quality.types import (
    ColumnDataType,
    ColumnProfile,
    NumericStats,
    TemporalStats,
)

_DEFAULT_TOP_K: int = 10


def profile_column(
    series: pd.Series,
    column_name: str,
    top_k: int = _DEFAULT_TOP_K,
) -> ColumnProfile:
    """Compute a full profile for a single column.

    Args:
        series: The pandas Series to profile.
        column_name: Name of the column (used for type inference and labeling).
        top_k: Number of most-frequent values to include.

    Returns:
        A frozen ``ColumnProfile`` attrs instance.
    """
    total_count: int = len(series)
    null_count: int = int(series.isna().sum())
    null_rate: float = null_count / total_count if total_count > 0 else 0.0

    non_null: pd.Series = series.dropna()
    distinct_count: int = int(non_null.nunique())
    non_null_count: int = len(non_null)
    cardinality_ratio: float = (
        distinct_count / non_null_count if non_null_count > 0 else 0.0
    )

    # Top-k values (by frequency, descending)
    top_values: list[tuple[str, int]] = []
    if non_null_count > 0:
        value_counts = non_null.value_counts().head(top_k)
        top_values = [(str(val), int(cnt)) for val, cnt in value_counts.items()]

    inferred_type: ColumnDataType = infer_column_type(series, column_name)

    # Compute optional stats based on inferred type
    numeric_stats: NumericStats | None = None
    temporal_stats: TemporalStats | None = None

    if inferred_type == ColumnDataType.NUMERIC or pd.api.types.is_numeric_dtype(
        series
    ):
        numeric_stats = compute_numeric_stats(non_null)

    if inferred_type == ColumnDataType.TEMPORAL or pd.api.types.is_datetime64_any_dtype(
        series
    ):
        temporal_stats = compute_temporal_stats(non_null)

    return ColumnProfile(
        column_name=column_name,
        inferred_type=inferred_type,
        total_count=total_count,
        null_count=null_count,
        null_rate=null_rate,
        distinct_count=distinct_count,
        cardinality_ratio=cardinality_ratio,
        top_values=top_values,
        numeric_stats=numeric_stats,
        temporal_stats=temporal_stats,
    )


def compute_numeric_stats(series: pd.Series) -> NumericStats:
    """Compute summary statistics for a numeric series.

    The series should already have NaN values removed.

    Args:
        series: A pandas Series of numeric values (NaN-free).

    Returns:
        A frozen ``NumericStats`` attrs instance.
    """
    if len(series) == 0:
        return NumericStats(
            mean=0.0,
            median=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
            p25=0.0,
            p75=0.0,
            skewness=None,
        )

    skewness_val: float | None = None
    if len(series) >= 3:
        skew_raw = series.skew()
        skewness_val = float(skew_raw) if pd.notna(skew_raw) else None  # type: ignore[arg-type]

    return NumericStats(
        mean=float(series.mean()),
        median=float(series.median()),
        std=float(series.std()),
        min_value=float(series.min()),
        max_value=float(series.max()),
        p25=float(series.quantile(0.25)),
        p75=float(series.quantile(0.75)),
        skewness=skewness_val,
    )


def compute_temporal_stats(series: pd.Series) -> TemporalStats:
    """Compute summary statistics for a datetime series.

    The series should already have NaT values removed.

    Args:
        series: A pandas Series of datetime values (NaT-free).

    Returns:
        A frozen ``TemporalStats`` attrs instance.
    """
    if len(series) == 0:
        sentinel = date(1970, 1, 1)
        return TemporalStats(
            min_date=sentinel,
            max_date=sentinel,
            date_range_days=0,
            most_common_day_of_week=None,
            most_common_month=None,
        )

    dt_series: pd.Series = pd.to_datetime(series)
    min_ts: pd.Timestamp = dt_series.min()
    max_ts: pd.Timestamp = dt_series.max()
    range_days: int = (max_ts - min_ts).days

    # Most common day of week (Monday=0 ... Sunday=6)
    day_of_week_counts = dt_series.dt.dayofweek.value_counts()
    most_common_dow: int = int(day_of_week_counts.index[0])

    # Most common month (1..12)
    month_counts = dt_series.dt.month.value_counts()
    most_common_month: int = int(month_counts.index[0])

    return TemporalStats(
        min_date=min_ts.date(),
        max_date=max_ts.date(),
        date_range_days=range_days,
        most_common_day_of_week=most_common_dow,
        most_common_month=most_common_month,
    )
