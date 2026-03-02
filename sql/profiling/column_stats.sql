-- Column-level statistics for a single column in a BigQuery table.
--
-- Parameters:
--   {table_name}  - fully-qualified BigQuery table (project.dataset.table)
--   {column_name} - name of the column to profile
--
-- Returns one row with null rate, distinct count, min, max, and average.

SELECT
    '{column_name}' AS column_name,
    COUNT(*) AS total_count,
    COUNTIF({column_name} IS NULL) AS null_count,
    SAFE_DIVIDE(COUNTIF({column_name} IS NULL), COUNT(*)) AS null_rate,
    COUNT(DISTINCT {column_name}) AS distinct_count,
    SAFE_DIVIDE(
        COUNT(DISTINCT {column_name}),
        COUNTIF({column_name} IS NOT NULL)
    ) AS cardinality_ratio,
    MIN({column_name}) AS min_value,
    MAX({column_name}) AS max_value,
    AVG(SAFE_CAST({column_name} AS FLOAT64)) AS avg_value
FROM
    `{table_name}`
