-- Period Comparison: Distribution Statistics for Two Time Periods
--
-- Computes descriptive statistics for a numeric column across two date-based
-- periods. Useful for detecting distributional drift in CJ data pipelines.
--
-- Parameters (replace with actual values):
--   @table       : Fully qualified table name
--   @date_col    : Date/timestamp column
--   @value_col   : Numeric column to compare
--   @period1_start: Start of period 1 (YYYY-MM-DD)
--   @period2_start: Start of period 2 (YYYY-MM-DD)
--
-- Example usage:
--   Replace placeholders below with your table and column names.

WITH period_data AS (
  SELECT
    CASE
      WHEN DATE(reporting_date) >= DATE('2023-01-01')
       AND DATE(reporting_date) < DATE('2023-07-01')
      THEN 'period_1'
      WHEN DATE(reporting_date) >= DATE('2023-07-01')
       AND DATE(reporting_date) < DATE('2024-01-01')
      THEN 'period_2'
    END AS period_label,
    total_population AS value
  FROM `project.dataset.population_metrics`
  WHERE DATE(reporting_date) >= DATE('2023-01-01')
    AND DATE(reporting_date) < DATE('2024-01-01')
    AND total_population IS NOT NULL
),

period_stats AS (
  SELECT
    period_label,
    COUNT(*) AS n,
    AVG(value) AS mean_value,
    STDDEV(value) AS stddev_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    APPROX_QUANTILES(value, 100)[OFFSET(25)] AS p25,
    APPROX_QUANTILES(value, 100)[OFFSET(50)] AS median_value,
    APPROX_QUANTILES(value, 100)[OFFSET(75)] AS p75
  FROM period_data
  WHERE period_label IS NOT NULL
  GROUP BY period_label
)

SELECT
  p1.period_label AS reference_period,
  p2.period_label AS comparison_period,
  p1.n AS ref_n,
  p2.n AS comp_n,
  p1.mean_value AS ref_mean,
  p2.mean_value AS comp_mean,
  ABS(p2.mean_value - p1.mean_value) AS mean_abs_change,
  CASE
    WHEN p1.stddev_value > 0
    THEN ABS(p2.mean_value - p1.mean_value) / p1.stddev_value
    ELSE NULL
  END AS mean_shift_in_stddevs,
  p1.stddev_value AS ref_stddev,
  p2.stddev_value AS comp_stddev,
  p1.median_value AS ref_median,
  p2.median_value AS comp_median,
  p1.p25 AS ref_p25,
  p2.p25 AS comp_p25,
  p1.p75 AS ref_p75,
  p2.p75 AS comp_p75
FROM period_stats p1
CROSS JOIN period_stats p2
WHERE p1.period_label = 'period_1'
  AND p2.period_label = 'period_2'
