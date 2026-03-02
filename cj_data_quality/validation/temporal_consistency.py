"""Temporal consistency checks for criminal justice date fields.

Validates date ordering (e.g., admission before release) and date
reasonableness (e.g., no dates before 1900 or in the far future).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cj_data_quality.constants import CJ_DATE_ORDERING_RULES, EARLIEST_REASONABLE_DATE


def check_date_ordering(
    df: pd.DataFrame,
    date_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Check that earlier_date <= later_date for each date pair.

    Args:
        df: Input DataFrame with datetime columns.
        date_pairs: List of (earlier_field, later_field) tuples to validate.
            Defaults to ``CJ_DATE_ORDERING_RULES``, filtered to columns
            actually present in *df*.

    Returns:
        DataFrame with columns: earlier_field, later_field, total_checked,
        violation_count, violation_rate.
    """
    if date_pairs is None:
        date_pairs = CJ_DATE_ORDERING_RULES

    results: list[dict[str, object]] = []

    for earlier_col, later_col in date_pairs:
        if earlier_col not in df.columns or later_col not in df.columns:
            continue

        mask_both_present = df[earlier_col].notna() & df[later_col].notna()
        total_checked = int(mask_both_present.sum())

        if total_checked == 0:
            results.append(
                {
                    "earlier_field": earlier_col,
                    "later_field": later_col,
                    "total_checked": 0,
                    "violation_count": 0,
                    "violation_rate": 0.0,
                }
            )
            continue

        subset = df.loc[mask_both_present]
        violations = subset[earlier_col] > subset[later_col]
        violation_count = int(violations.sum())

        results.append(
            {
                "earlier_field": earlier_col,
                "later_field": later_col,
                "total_checked": total_checked,
                "violation_count": violation_count,
                "violation_rate": violation_count / total_checked,
            }
        )

    return pd.DataFrame(results)


def check_date_reasonableness(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
    earliest: str = EARLIEST_REASONABLE_DATE,
    latest: str | None = None,
) -> pd.DataFrame:
    """Flag dates before *earliest* or after *latest* (default = today).

    Args:
        df: Input DataFrame with datetime columns.
        date_columns: Columns to check.  When ``None``, all datetime columns
            in *df* are checked.
        earliest: ISO-format date string for the lower bound.
        latest: ISO-format date string for the upper bound.  Defaults to
            today's date.

    Returns:
        DataFrame with columns: column, total_checked, before_earliest_count,
        after_latest_count, unreasonable_rate.
    """
    if latest is None:
        latest = date.today().isoformat()

    earliest_ts = pd.Timestamp(earliest)
    latest_ts = pd.Timestamp(latest)

    if date_columns is None:
        date_columns = [
            col
            for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]

    results: list[dict[str, object]] = []

    for col in date_columns:
        if col not in df.columns:
            continue

        non_null = df[col].dropna()
        total_checked = len(non_null)

        if total_checked == 0:
            results.append(
                {
                    "column": col,
                    "total_checked": 0,
                    "before_earliest_count": 0,
                    "after_latest_count": 0,
                    "unreasonable_rate": 0.0,
                }
            )
            continue

        before_earliest = int((non_null < earliest_ts).sum())
        after_latest = int((non_null > latest_ts).sum())
        unreasonable = before_earliest + after_latest

        results.append(
            {
                "column": col,
                "total_checked": total_checked,
                "before_earliest_count": before_earliest,
                "after_latest_count": after_latest,
                "unreasonable_rate": unreasonable / total_checked,
            }
        )

    return pd.DataFrame(results)


def find_date_violations(
    df: pd.DataFrame,
    earlier_col: str,
    later_col: str,
) -> pd.DataFrame:
    """Return the rows where *earlier_col* > *later_col*.

    Args:
        df: Input DataFrame.
        earlier_col: Column name expected to contain the earlier date.
        later_col: Column name expected to contain the later date.

    Returns:
        A DataFrame containing only the violating rows (preserving the
        original index).
    """
    mask_both_present = df[earlier_col].notna() & df[later_col].notna()
    subset = df.loc[mask_both_present]
    violations = subset[earlier_col] > subset[later_col]
    return df.loc[violations[violations].index]
