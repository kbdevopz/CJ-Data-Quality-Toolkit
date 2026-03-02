"""Coverage matrix: state x metric completeness analysis.

Builds a pivot matrix showing data completeness (1 - null_rate) for each
state-metric pair, identifies gaps below a threshold, and summarizes
coverage by metric.
"""

from __future__ import annotations

import pandas as pd

from cj_data_quality.types import CoverageCell


def build_coverage_matrix(
    df: pd.DataFrame,
    state_col: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Build a coverage matrix with states as rows and metrics as columns.

    Each cell value is the completeness ratio (0.0 to 1.0) defined as
    ``1 - null_rate`` for that state-metric pair.

    Args:
        df: Input DataFrame containing state and metric columns.
        state_col: Column name identifying the state.
        metric_cols: List of metric column names to evaluate.

    Returns:
        A DataFrame indexed by state code with one column per metric,
        values representing completeness (0.0 to 1.0).
    """
    records: list[dict[str, object]] = []
    total_counts: dict[tuple[str, str], int] = {}
    present_counts: dict[tuple[str, str], int] = {}

    for state, group in df.groupby(state_col):
        row: dict[str, object] = {state_col: state}
        for metric in metric_cols:
            total = len(group)
            present = int(group[metric].notna().sum())
            completeness = float(present / total) if total > 0 else 0.0
            row[metric] = completeness
            total_counts[(str(state), metric)] = total
            present_counts[(str(state), metric)] = present
        records.append(row)

    matrix = pd.DataFrame(records).set_index(state_col)
    matrix.index.name = state_col
    matrix.attrs["_total_counts"] = total_counts
    matrix.attrs["_present_counts"] = present_counts
    return matrix


def identify_coverage_gaps(
    matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> list[CoverageCell]:
    """Identify cells in the coverage matrix that fall below a threshold.

    Args:
        matrix: Coverage matrix produced by :func:`build_coverage_matrix`.
        threshold: Minimum acceptable completeness (default 0.8).

    Returns:
        List of :class:`CoverageCell` instances for every cell whose
        completeness is strictly below *threshold*, sorted by completeness
        ascending.
    """
    gaps: list[CoverageCell] = []
    total_counts: dict[tuple[str, str], int] = matrix.attrs.get(
        "_total_counts", {}
    )
    present_counts: dict[tuple[str, str], int] = matrix.attrs.get(
        "_present_counts", {}
    )

    for state in matrix.index:
        for metric in matrix.columns:
            completeness = float(matrix.loc[state, metric])
            if completeness < threshold:
                key = (str(state), str(metric))
                total_expected = total_counts.get(key, 0)
                total_present = present_counts.get(key, 0)
                gaps.append(
                    CoverageCell(
                        state_code=str(state),
                        metric_name=str(metric),
                        completeness=completeness,
                        total_expected=total_expected,
                        total_present=total_present,
                        is_gap=True,
                    )
                )

    gaps.sort(key=lambda c: c.completeness)
    return gaps


def summarize_coverage(matrix: pd.DataFrame) -> dict[str, float]:
    """Compute mean completeness per metric across all states.

    Args:
        matrix: Coverage matrix produced by :func:`build_coverage_matrix`.

    Returns:
        Dictionary mapping each metric name to its mean completeness
        (0.0 to 1.0) across states.
    """
    return {col: float(matrix[col].mean()) for col in matrix.columns}
