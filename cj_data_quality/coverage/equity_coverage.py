"""Equity coverage analysis: demographic completeness and disparity indices.

Provides functions to measure how completely demographic fields are reported
across states, and to compute within-state disparity indices that quantify
how unevenly a metric is distributed across demographic groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cj_data_quality.constants import CJ_DEMOGRAPHIC_FIELDS
from cj_data_quality.types import EquityCoverage


def analyze_demographic_completeness(
    df: pd.DataFrame,
    state_col: str = "state_code",
    demographic_fields: list[str] | None = None,
) -> list[EquityCoverage]:
    """Analyse demographic field completeness per state.

    For each state and demographic field, computes:
    - completeness (1 - null_rate)
    - distinct non-null values
    - most common value and its rate

    Args:
        df: Input DataFrame.
        state_col: Column identifying the state.
        demographic_fields: Demographic columns to evaluate.  Defaults to
            :data:`CJ_DEMOGRAPHIC_FIELDS` filtered to columns present in *df*.

    Returns:
        List of :class:`EquityCoverage` instances, one per state-field pair.
    """
    if demographic_fields is None:
        demographic_fields = [
            f for f in CJ_DEMOGRAPHIC_FIELDS if f in df.columns
        ]

    results: list[EquityCoverage] = []

    for state, group in df.groupby(state_col):
        for field in demographic_fields:
            if field not in df.columns:
                continue

            total = len(group)
            non_null = group[field].dropna()
            present = len(non_null)
            completeness = float(present / total) if total > 0 else 0.0
            distinct_values = int(non_null.nunique())

            most_common_value: str | None = None
            most_common_rate: float | None = None

            if present > 0:
                value_counts = non_null.value_counts()
                most_common_value = str(value_counts.index[0])
                most_common_rate = float(value_counts.iloc[0] / present)

            results.append(
                EquityCoverage(
                    state_code=str(state),
                    field_name=field,
                    completeness=completeness,
                    distinct_values=distinct_values,
                    most_common_value=most_common_value,
                    most_common_rate=most_common_rate,
                )
            )

    return results


def compute_equity_disparity_index(
    df: pd.DataFrame,
    state_col: str,
    group_col: str,
    metric_col: str,
) -> dict[str, float]:
    """Compute a per-state disparity index for a metric across demographic groups.

    The disparity index is the **coefficient of variation** (std / mean) of the
    metric values across the distinct groups within each state.  A higher value
    indicates more inequality across groups; zero means all groups have
    identical metric values.

    Only non-null rows are considered.  States or groups with insufficient data
    (fewer than 2 groups with non-null metric values) are assigned a disparity
    index of 0.0.

    Args:
        df: Input DataFrame.
        state_col: Column identifying the state.
        group_col: Column identifying the demographic group (e.g. ``"race"``).
        metric_col: Column with the numeric metric to measure (e.g. ``"age"``).

    Returns:
        Dictionary mapping state codes to their disparity index (float).
    """
    result: dict[str, float] = {}

    filtered = df[[state_col, group_col, metric_col]].dropna()

    for state, state_group in filtered.groupby(state_col):
        group_means = state_group.groupby(group_col)[metric_col].mean()

        if len(group_means) < 2:
            result[str(state)] = 0.0
            continue

        mean_val = float(group_means.mean())
        std_val = float(group_means.std(ddof=0))

        if mean_val == 0.0:
            result[str(state)] = 0.0
        else:
            result[str(state)] = std_val / mean_val

    return result
