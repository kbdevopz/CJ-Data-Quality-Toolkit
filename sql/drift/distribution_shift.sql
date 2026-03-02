-- Distribution Shift Detection Using LAG-Based Quarterly Comparison
--
-- Computes quarterly statistics for a numeric column and compares each
-- quarter to the previous one using LAG window functions. Flags quarters
-- where the mean shifts by more than 2 standard deviations of the prior
-- quarter, indicating potential distributional drift.
--
-- Parameters (replace with actual values):
--   @table     : Fully qualified table name
--   @date_col  : Date/timestamp column
--   @value_col : Numeric column to monitor
--
-- Example usage:
--   Replace placeholders below with your table and column names.

WITH quarterly_stats AS (
  SELECT
    DATE_TRUNC(DATE(reporting_date), QUARTER) AS quarter_start,
    COUNT(*) AS n,
    AVG(total_population) AS mean_value,
    STDDEV(total_population) AS stddev_value,
    MIN(total_population) AS min_value,
    MAX(total_population) AS max_value,
    APPROX_QUANTILES(total_population, 100)[OFFSET(25)] AS p25,
    APPROX_QUANTILES(total_population, 100)[OFFSET(50)] AS median_value,
    APPROX_QUANTILES(total_population, 100)[OFFSET(75)] AS p75
  FROM `project.dataset.population_metrics`
  WHERE total_population IS NOT NULL
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
  SAFE_DIVIDE(
    ABS(mean_value - prev_mean),
    prev_stddev
  ) AS mean_shift_in_stddevs,
  CASE
    WHEN prev_stddev > 0
     AND ABS(mean_value - prev_mean) / prev_stddev > 2.0
    THEN 'DRIFT_DETECTED'
    WHEN prev_stddev > 0
     AND ABS(mean_value - prev_mean) / prev_stddev > 1.0
    THEN 'WARNING'
    ELSE 'STABLE'
  END AS drift_flag,
  CASE
    WHEN prev_mean > 0
    THEN (mean_value - prev_mean) / prev_mean * 100
    ELSE NULL
  END AS pct_change
FROM shift_detection
WHERE prev_quarter IS NOT NULL
ORDER BY quarter_start
