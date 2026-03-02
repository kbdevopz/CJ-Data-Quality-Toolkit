"""Generate BigQuery-compatible SQL for profiling operations.

Each function returns a SQL string with table/column names interpolated.
These mirror the standalone SQL files in ``sql/profiling/`` but are
generated programmatically for use in Python pipelines.
"""

import textwrap


def generate_column_stats_sql(table: str, column: str) -> str:
    """Generate BigQuery SQL for basic column statistics.

    Computes null rate, distinct count, min, max, and average (for numeric
    columns) in a single query.

    Args:
        table: Fully-qualified BigQuery table name.
        column: Column name to profile.

    Returns:
        A BigQuery SQL string.
    """
    return textwrap.dedent(f"""\
        SELECT
            '{column}' AS column_name,
            COUNT(*) AS total_count,
            COUNTIF({column} IS NULL) AS null_count,
            SAFE_DIVIDE(COUNTIF({column} IS NULL), COUNT(*)) AS null_rate,
            COUNT(DISTINCT {column}) AS distinct_count,
            SAFE_DIVIDE(COUNT(DISTINCT {column}), COUNTIF({column} IS NOT NULL)) AS cardinality_ratio,
            MIN({column}) AS min_value,
            MAX({column}) AS max_value,
            AVG(SAFE_CAST({column} AS FLOAT64)) AS avg_value
        FROM
            `{table}`
    """)


def generate_null_rate_sql(table: str, columns: list[str] | None = None) -> str:
    """Generate BigQuery SQL for null rate across specified columns.

    When *columns* is provided the query uses ``UNPIVOT`` with a static
    column list (required by BigQuery).  When omitted a per-column
    ``COUNTIF`` approach is generated instead.

    Args:
        table: Fully-qualified BigQuery table name.
        columns: Optional list of column names to check.

    Returns:
        A BigQuery SQL string.
    """
    if columns:
        col_list = ", ".join(f"`{c}`" for c in columns)
        return textwrap.dedent(f"""\
            SELECT
                column_name,
                COUNTIF(value IS NULL) AS null_count,
                COUNT(*) AS total_count,
                SAFE_DIVIDE(COUNTIF(value IS NULL), COUNT(*)) AS null_rate
            FROM
                `{table}`
            UNPIVOT(
                value FOR column_name IN ({col_list})
            )
            GROUP BY
                column_name
            ORDER BY
                null_rate DESC
        """)

    # Fallback without UNPIVOT — caller should supply column names
    return textwrap.dedent(f"""\
        -- NOTE: Pass column names to generate_null_rate_sql() for a
        -- working UNPIVOT query. This template requires manual editing.
        SELECT
            column_name,
            null_count,
            total_count,
            SAFE_DIVIDE(null_count, total_count) AS null_rate
        FROM (
            SELECT '{{COLUMN}}' AS column_name,
                   COUNTIF(`{{COLUMN}}` IS NULL) AS null_count,
                   COUNT(*) AS total_count
            FROM `{table}`
        )
        ORDER BY null_rate DESC
    """)


def generate_distribution_sql(table: str, column: str) -> str:
    """Generate BigQuery SQL for percentile distribution of a numeric column.

    Computes p5, p25, p50 (median), p75, p95 using ``APPROX_QUANTILES``,
    along with mean, standard deviation, and skewness approximation.

    Args:
        table: Fully-qualified BigQuery table name.
        column: Numeric column name.

    Returns:
        A BigQuery SQL string.
    """
    return textwrap.dedent(f"""\
        SELECT
            '{column}' AS column_name,
            COUNT(*) AS total_count,
            COUNTIF({column} IS NOT NULL) AS non_null_count,
            AVG({column}) AS mean_value,
            STDDEV({column}) AS std_value,
            MIN({column}) AS min_value,
            MAX({column}) AS max_value,
            percentiles[OFFSET(5)] AS p5,
            percentiles[OFFSET(25)] AS p25,
            percentiles[OFFSET(50)] AS p50_median,
            percentiles[OFFSET(75)] AS p75,
            percentiles[OFFSET(95)] AS p95
        FROM
            `{table}`,
            UNNEST([STRUCT(
                APPROX_QUANTILES({column}, 100) AS percentiles
            )])
        WHERE
            {column} IS NOT NULL
    """)
