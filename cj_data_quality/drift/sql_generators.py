"""SQL generators for distribution drift analysis in BigQuery.

Produces parameterized BigQuery SQL for period-over-period comparisons
and LAG-based distribution shift detection.
"""

from __future__ import annotations


def generate_period_comparison_sql(
    table: str,
    date_col: str,
    value_col: str,
    period1: str,
    period2: str,
) -> str:
    """Generate BigQuery SQL to compare distributions between two date periods.

    The query computes descriptive statistics (count, mean, stddev, min, max,
    and selected percentiles) for each period and unions them for comparison.

    Args:
        table: Fully qualified BigQuery table name (e.g. ``"project.dataset.table"``).
        date_col: Name of the date/timestamp column.
        value_col: Name of the numeric column to compare.
        period1: Start date of the first period in ``"YYYY-MM-DD"`` format.
        period2: Start date of the second period in ``"YYYY-MM-DD"`` format.

    Returns:
        A BigQuery SQL string.
    """
    return f"""\
-- Period comparison: distribution statistics for two periods
-- Table: {table}
-- Periods: {period1} vs {period2}

WITH period_data AS (
  SELECT
    CASE
      WHEN DATE({date_col}) >= DATE('{period1}')
       AND DATE({date_col}) < DATE('{period2}')
      THEN 'period_1'
      WHEN DATE({date_col}) >= DATE('{period2}')
      THEN 'period_2'
    END AS period_label,
    {value_col}
  FROM `{table}`
  WHERE DATE({date_col}) >= DATE('{period1}')
    AND {value_col} IS NOT NULL
)

SELECT
  period_label,
  COUNT(*) AS n,
  AVG({value_col}) AS mean_value,
  STDDEV({value_col}) AS stddev_value,
  MIN({value_col}) AS min_value,
  MAX({value_col}) AS max_value,
  APPROX_QUANTILES({value_col}, 100)[OFFSET(25)] AS p25,
  APPROX_QUANTILES({value_col}, 100)[OFFSET(50)] AS median_value,
  APPROX_QUANTILES({value_col}, 100)[OFFSET(75)] AS p75
FROM period_data
WHERE period_label IS NOT NULL
GROUP BY period_label
ORDER BY period_label
"""


def generate_distribution_shift_sql(
    table: str,
    date_col: str,
    value_col: str,
) -> str:
    """Generate BigQuery SQL for LAG-based distribution shift detection.

    Computes quarterly statistics and compares each quarter to the previous
    one using LAG window functions. Flags quarters where the mean shifts
    by more than 2 standard deviations of the prior quarter.

    Args:
        table: Fully qualified BigQuery table name.
        date_col: Name of the date/timestamp column.
        value_col: Name of the numeric column to monitor.

    Returns:
        A BigQuery SQL string.
    """
    return f"""\
-- Distribution shift detection using LAG-based comparison
-- Table: {table}
-- Column: {value_col} by quarter

WITH quarterly_stats AS (
  SELECT
    DATE_TRUNC(DATE({date_col}), QUARTER) AS quarter_start,
    COUNT(*) AS n,
    AVG({value_col}) AS mean_value,
    STDDEV({value_col}) AS stddev_value,
    MIN({value_col}) AS min_value,
    MAX({value_col}) AS max_value,
    APPROX_QUANTILES({value_col}, 100)[OFFSET(50)] AS median_value
  FROM `{table}`
  WHERE {value_col} IS NOT NULL
  GROUP BY quarter_start
),

shift_detection AS (
  SELECT
    quarter_start,
    n,
    mean_value,
    stddev_value,
    median_value,
    LAG(mean_value) OVER (ORDER BY quarter_start) AS prev_mean,
    LAG(stddev_value) OVER (ORDER BY quarter_start) AS prev_stddev,
    LAG(n) OVER (ORDER BY quarter_start) AS prev_n,
    LAG(quarter_start) OVER (ORDER BY quarter_start) AS prev_quarter
  FROM quarterly_stats
)

SELECT
  quarter_start,
  prev_quarter,
  n,
  mean_value,
  prev_mean,
  stddev_value,
  prev_stddev,
  ABS(mean_value - prev_mean) AS mean_abs_change,
  CASE
    WHEN prev_stddev > 0
    THEN ABS(mean_value - prev_mean) / prev_stddev
    ELSE NULL
  END AS mean_shift_in_stddevs,
  CASE
    WHEN prev_stddev > 0
     AND ABS(mean_value - prev_mean) / prev_stddev > 2.0
    THEN 'DRIFT_DETECTED'
    ELSE 'STABLE'
  END AS drift_flag
FROM shift_detection
WHERE prev_quarter IS NOT NULL
ORDER BY quarter_start
"""
