"""Referential integrity checks across criminal justice data tables.

Validates foreign-key relationships (child values exist in parent) and
measures overlap between tables on shared join columns.
"""

from __future__ import annotations

import pandas as pd


def check_foreign_key(
    child_df: pd.DataFrame,
    child_col: str,
    parent_df: pd.DataFrame,
    parent_col: str,
) -> dict[str, int | float | list[object]]:
    """Check that all child values exist in the parent table.

    Args:
        child_df: DataFrame containing the foreign key column.
        child_col: Column name in *child_df*.
        parent_df: DataFrame containing the referenced primary key.
        parent_col: Column name in *parent_df*.

    Returns:
        Dictionary with keys: total_child_values, orphan_count,
        orphan_rate, sample_orphans (list of up to 10 orphan values).
    """
    child_values = child_df[child_col].dropna()
    parent_values = set(parent_df[parent_col].dropna())

    total_child_values = len(child_values)
    orphans = child_values[~child_values.isin(parent_values)]
    orphan_count = len(orphans)
    orphan_rate = orphan_count / total_child_values if total_child_values > 0 else 0.0

    sample_orphans = orphans.unique().tolist()[:10]

    return {
        "total_child_values": total_child_values,
        "orphan_count": orphan_count,
        "orphan_rate": orphan_rate,
        "sample_orphans": sample_orphans,
    }


def check_cross_table_consistency(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    join_col: str,
) -> dict[str, int | float]:
    """Check overlap between two tables on a shared join column.

    Args:
        df_a: First DataFrame.
        df_b: Second DataFrame.
        join_col: Column name present in both DataFrames.

    Returns:
        Dictionary with keys: a_only_count, b_only_count, both_count,
        overlap_rate (fraction of the union that appears in both).
    """
    values_a = set(df_a[join_col].dropna())
    values_b = set(df_b[join_col].dropna())

    both = values_a & values_b
    a_only = values_a - values_b
    b_only = values_b - values_a
    union_size = len(values_a | values_b)

    return {
        "a_only_count": len(a_only),
        "b_only_count": len(b_only),
        "both_count": len(both),
        "overlap_rate": len(both) / union_size if union_size > 0 else 0.0,
    }
