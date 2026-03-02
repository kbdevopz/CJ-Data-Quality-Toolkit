"""Table-level profiling: aggregate column profiles into a single TableProfile.

Computes per-column profiles, duplicate row counts, and overall null rates.
"""

import pandas as pd

from cj_data_quality.profiling.column_profiler import profile_column
from cj_data_quality.types import ColumnProfile, TableProfile


def profile_table(
    df: pd.DataFrame,
    table_name: str,
) -> TableProfile:
    """Profile an entire DataFrame, producing a ``TableProfile``.

    Args:
        df: The DataFrame to profile.
        table_name: A human-readable name for the table.

    Returns:
        A frozen ``TableProfile`` attrs instance containing per-column
        profiles, duplicate statistics, and the overall null rate.
    """
    row_count: int = len(df)
    column_count: int = len(df.columns)

    # Profile each column
    column_profiles: list[ColumnProfile] = [
        profile_column(df[col], col) for col in df.columns
    ]

    # Duplicate detection
    duplicate_row_count: int = int(df.duplicated().sum())
    duplicate_rate: float = (
        duplicate_row_count / row_count if row_count > 0 else 0.0
    )

    # Overall null rate: total nulls across all cells / total cells
    total_cells: int = row_count * column_count
    total_nulls: int = int(df.isna().sum().sum())
    overall_null_rate: float = total_nulls / total_cells if total_cells > 0 else 0.0

    return TableProfile(
        table_name=table_name,
        row_count=row_count,
        column_count=column_count,
        column_profiles=column_profiles,
        duplicate_row_count=duplicate_row_count,
        duplicate_rate=duplicate_rate,
        overall_null_rate=overall_null_rate,
    )
