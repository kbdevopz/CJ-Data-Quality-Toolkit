"""SQL generators for anomaly detection queries targeting BigQuery.

Generates parameterised BigQuery-compatible SQL for Z-score detection and
rolling-window statistics.
"""


def generate_zscore_detection_sql(
    table: str,
    date_col: str,
    value_col: str,
    threshold: float,
) -> str:
    """Generate BigQuery SQL that flags rows with |Z-score| above *threshold*.

    The query computes the population mean and standard deviation across the
    entire table, then returns rows whose absolute Z-score exceeds the
    supplied threshold.

    Args:
        table: Fully-qualified BigQuery table name (e.g.
            ``project.dataset.table``).
        date_col: Name of the date column.
        value_col: Name of the numeric value column.
        threshold: Absolute Z-score cutoff.

    Returns:
        A BigQuery SQL string.
    """
    return f"""\
WITH stats AS (
    SELECT
        AVG({value_col}) AS mean_val,
        STDDEV_POP({value_col}) AS std_val
    FROM `{table}`
),
scored AS (
    SELECT
        t.{date_col},
        t.{value_col},
        s.mean_val,
        s.std_val,
        SAFE_DIVIDE(t.{value_col} - s.mean_val, s.std_val) AS zscore
    FROM `{table}` t
    CROSS JOIN stats s
)
SELECT
    {date_col},
    {value_col},
    mean_val,
    std_val,
    zscore,
    ABS(zscore) AS abs_zscore
FROM scored
WHERE ABS(zscore) > {threshold}
ORDER BY ABS(zscore) DESC"""


def generate_rolling_stats_sql(
    table: str,
    date_col: str,
    value_col: str,
    window: int,
) -> str:
    """Generate BigQuery SQL that computes rolling mean and std deviation.

    Uses ``ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW`` to compute
    a trailing rolling window.

    Args:
        table: Fully-qualified BigQuery table name.
        date_col: Name of the date column.
        value_col: Name of the numeric value column.
        window: Rolling window size in rows.

    Returns:
        A BigQuery SQL string.
    """
    preceding: int = window - 1
    return f"""\
SELECT
    {date_col},
    {value_col},
    AVG({value_col}) OVER (
        ORDER BY {date_col}
        ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
    ) AS rolling_mean,
    STDDEV_POP({value_col}) OVER (
        ORDER BY {date_col}
        ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
    ) AS rolling_std,
    COUNT({value_col}) OVER (
        ORDER BY {date_col}
        ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
    ) AS window_count
FROM `{table}`
ORDER BY {date_col}"""
