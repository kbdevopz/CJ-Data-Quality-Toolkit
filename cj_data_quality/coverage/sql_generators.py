"""SQL generators for coverage analysis (BigQuery dialect).

Each function returns a SQL string suitable for execution on BigQuery.
"""

from __future__ import annotations


def generate_cross_state_matrix_sql(
    table: str,
    state_col: str,
    metric_cols: list[str],
) -> str:
    """Generate BigQuery SQL for a cross-state coverage matrix.

    The resulting query produces one row per state with completeness
    (1 - null_rate) for every metric column.

    Args:
        table: Fully-qualified BigQuery table reference
               (e.g. ``"project.dataset.table"``).
        state_col: Column identifying the state.
        metric_cols: Metric columns to include in the matrix.

    Returns:
        A BigQuery-compatible SQL string.
    """
    completeness_exprs = ",\n    ".join(
        f"1.0 - COUNTIF({col} IS NULL) / COUNT(*) AS {col}_completeness"
        for col in metric_cols
    )

    sql = (
        f"SELECT\n"
        f"    {state_col},\n"
        f"    COUNT(*) AS total_rows,\n"
        f"    {completeness_exprs}\n"
        f"FROM\n"
        f"    `{table}`\n"
        f"GROUP BY\n"
        f"    {state_col}\n"
        f"ORDER BY\n"
        f"    {state_col}"
    )
    return sql


def generate_demographic_completeness_sql(
    table: str,
    state_col: str,
    demo_col: str,
) -> str:
    """Generate BigQuery SQL for demographic field completeness by state.

    The query returns, per state:
    - total row count
    - non-null count and completeness for the demographic column
    - distinct non-null value count
    - most common value and its rate (using APPROX_TOP_COUNT)

    Args:
        table: Fully-qualified BigQuery table reference.
        state_col: Column identifying the state.
        demo_col: Demographic column to analyse.

    Returns:
        A BigQuery-compatible SQL string.
    """
    sql = (
        f"WITH base AS (\n"
        f"    SELECT\n"
        f"        {state_col},\n"
        f"        {demo_col},\n"
        f"        COUNT(*) OVER (PARTITION BY {state_col}) AS total_rows,\n"
        f"        COUNT({demo_col}) OVER (PARTITION BY {state_col}) AS non_null_count\n"
        f"    FROM\n"
        f"        `{table}`\n"
        f"),\n"
        f"completeness AS (\n"
        f"    SELECT\n"
        f"        {state_col},\n"
        f"        ANY_VALUE(total_rows) AS total_rows,\n"
        f"        ANY_VALUE(non_null_count) AS non_null_count,\n"
        f"        SAFE_DIVIDE(ANY_VALUE(non_null_count), ANY_VALUE(total_rows)) AS completeness,\n"
        f"        COUNT(DISTINCT {demo_col}) AS distinct_values\n"
        f"    FROM\n"
        f"        base\n"
        f"    GROUP BY\n"
        f"        {state_col}\n"
        f"),\n"
        f"most_common AS (\n"
        f"    SELECT\n"
        f"        {state_col},\n"
        f"        {demo_col} AS most_common_value,\n"
        f"        COUNT(*) AS value_count,\n"
        f"        ROW_NUMBER() OVER (PARTITION BY {state_col} ORDER BY COUNT(*) DESC) AS rn\n"
        f"    FROM\n"
        f"        `{table}`\n"
        f"    WHERE\n"
        f"        {demo_col} IS NOT NULL\n"
        f"    GROUP BY\n"
        f"        {state_col}, {demo_col}\n"
        f")\n"
        f"SELECT\n"
        f"    c.{state_col},\n"
        f"    c.total_rows,\n"
        f"    c.non_null_count,\n"
        f"    c.completeness,\n"
        f"    c.distinct_values,\n"
        f"    m.most_common_value,\n"
        f"    SAFE_DIVIDE(m.value_count, c.non_null_count) AS most_common_rate\n"
        f"FROM\n"
        f"    completeness c\n"
        f"LEFT JOIN\n"
        f"    most_common m\n"
        f"ON\n"
        f"    c.{state_col} = m.{state_col} AND m.rn = 1\n"
        f"ORDER BY\n"
        f"    c.{state_col}"
    )
    return sql
