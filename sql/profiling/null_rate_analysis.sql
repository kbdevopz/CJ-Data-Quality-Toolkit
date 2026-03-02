-- Null rate analysis across all columns in a BigQuery table.
--
-- Parameters:
--   {table_name} - fully-qualified BigQuery table (project.dataset.table)
--
-- Returns one row per column with null count, total count, and null rate,
-- ordered by null rate descending (worst columns first).

SELECT
    column_name,
    COUNTIF(value IS NULL) AS null_count,
    COUNT(*) AS total_count,
    SAFE_DIVIDE(COUNTIF(value IS NULL), COUNT(*)) AS null_rate
FROM
    `{table_name}`
UNPIVOT(
    value FOR column_name IN (
        SELECT column_name
        FROM `{table_name}`.INFORMATION_SCHEMA.COLUMNS
    )
)
GROUP BY
    column_name
ORDER BY
    null_rate DESC
