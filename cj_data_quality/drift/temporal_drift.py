"""Temporal drift detection across time periods.

Splits data by calendar period (e.g. quarters) and runs Kolmogorov-Smirnov
tests on consecutive periods to detect distributional shifts over time.
"""

from __future__ import annotations

import pandas as pd

from cj_data_quality.drift.distribution_drift import detect_numeric_drift
from cj_data_quality.types import DriftResult, DriftSeverity


def detect_temporal_drift(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: str = "Q",
    group_col: str | None = None,
) -> list[DriftResult]:
    """Detect distributional drift across consecutive time periods.

    Splits the data by ``period`` (using pandas period notation), then runs
    a Kolmogorov-Smirnov two-sample test between every pair of consecutive
    periods. If ``group_col`` is provided, runs the analysis independently
    for each group.

    Args:
        df: Input DataFrame containing at least ``date_col`` and ``value_col``.
        date_col: Name of the datetime column used to assign periods.
        value_col: Name of the numeric column to test for drift.
        period: Pandas period alias (e.g. ``"Q"`` for quarter, ``"M"`` for
            month, ``"Y"`` for year). Defaults to ``"Q"``.
        group_col: Optional column name for per-group analysis (e.g.
            ``"state_code"``).

    Returns:
        A list of DriftResult objects, one per consecutive period pair
        (and per group, if ``group_col`` is specified). Empty list if
        fewer than two periods exist.

    Raises:
        KeyError: If any of the specified columns do not exist in ``df``.
    """
    for col in [date_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    if group_col is not None and group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame.")

    # Work on a copy to avoid mutating the input
    work_df: pd.DataFrame = df[[date_col, value_col] + ([group_col] if group_col else [])].copy()
    work_df = work_df.dropna(subset=[date_col, value_col])

    # Ensure date column is datetime
    work_df[date_col] = pd.to_datetime(work_df[date_col])

    # Assign period labels
    work_df["_period"] = work_df[date_col].dt.to_period(period)

    results: list[DriftResult] = []

    if group_col is not None:
        groups = sorted(work_df[group_col].unique())
        for group_value in groups:
            group_df: pd.DataFrame = work_df[work_df[group_col] == group_value]
            group_results = _run_consecutive_ks(
                group_df, value_col, column_label=f"{value_col}[{group_value}]"
            )
            results.extend(group_results)
    else:
        results = _run_consecutive_ks(
            work_df, value_col, column_label=value_col
        )

    return results


def _run_consecutive_ks(
    df: pd.DataFrame,
    value_col: str,
    column_label: str,
) -> list[DriftResult]:
    """Run KS tests on consecutive period pairs within a single group.

    Args:
        df: DataFrame with ``_period`` and ``value_col`` columns.
        value_col: The numeric column to test.
        column_label: Label for the column (may include group info).

    Returns:
        List of DriftResult for each consecutive period pair.
    """
    periods = sorted(df["_period"].unique())
    results: list[DriftResult] = []

    for i in range(len(periods) - 1):
        ref_period = periods[i]
        comp_period = periods[i + 1]

        ref_data: pd.Series = df.loc[df["_period"] == ref_period, value_col]
        comp_data: pd.Series = df.loc[df["_period"] == comp_period, value_col]

        # Skip if either period has no data
        if len(ref_data) == 0 or len(comp_data) == 0:
            continue

        result: DriftResult = detect_numeric_drift(
            reference=ref_data,
            comparison=comp_data,
            column_name=column_label,
            ref_period=str(ref_period),
            comp_period=str(comp_period),
        )
        results.append(result)

    return results


def summarize_drift_over_time(results: list[DriftResult]) -> pd.DataFrame:
    """Create a summary pivot table of drift results over time.

    Returns a DataFrame with columns:
        - column_name: The column (or column[group]) tested.
        - reference_period: The earlier period label.
        - comparison_period: The later period label.
        - statistic: The test statistic (KS D or chi-squared).
        - p_value: The p-value from the test.
        - severity: The drift severity label.

    Args:
        results: A list of DriftResult objects.

    Returns:
        A pandas DataFrame summarizing drift over time. Empty DataFrame
        with the expected columns if the input list is empty.
    """
    columns: list[str] = [
        "column_name",
        "reference_period",
        "comparison_period",
        "statistic",
        "p_value",
        "severity",
    ]

    if not results:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for r in results:
        rows.append(
            {
                "column_name": r.column_name,
                "reference_period": r.reference_period,
                "comparison_period": r.comparison_period,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "severity": r.severity.value,
            }
        )

    return pd.DataFrame(rows, columns=columns)
