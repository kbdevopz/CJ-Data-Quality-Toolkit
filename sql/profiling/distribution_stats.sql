-- Distribution statistics (percentiles) for a numeric column.
--
-- Parameters:
--   {table_name}  - fully-qualified BigQuery table (project.dataset.table)
--   {column_name} - name of the numeric column to profile
--
-- Returns one row with count, mean, std, min, max, and percentiles
-- (p5, p25, p50/median, p75, p95).

SELECT
    '{column_name}' AS column_name,
    COUNT(*) AS total_count,
    COUNTIF({column_name} IS NOT NULL) AS non_null_count,
    AVG({column_name}) AS mean_value,
    STDDEV({column_name}) AS std_value,
    MIN({column_name}) AS min_value,
    MAX({column_name}) AS max_value,
    percentiles[OFFSET(5)] AS p5,
    percentiles[OFFSET(25)] AS p25,
    percentiles[OFFSET(50)] AS p50_median,
    percentiles[OFFSET(75)] AS p75,
    percentiles[OFFSET(95)] AS p95
FROM
    `{table_name}`,
    UNNEST([STRUCT(
        APPROX_QUANTILES({column_name}, 100) AS percentiles
    )])
WHERE
    {column_name} IS NOT NULL
