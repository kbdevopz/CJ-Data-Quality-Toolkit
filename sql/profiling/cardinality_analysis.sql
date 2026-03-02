-- Cardinality analysis for a single column in a BigQuery table.
--
-- Parameters:
--   {table_name}  - fully-qualified BigQuery table (project.dataset.table)
--   {column_name} - name of the column to analyze
--
-- Returns the distinct count, non-null count, cardinality ratio,
-- and the top 20 most frequent values with their counts.

WITH cardinality AS (
    SELECT
        '{column_name}' AS column_name,
        COUNT(DISTINCT {column_name}) AS distinct_count,
        COUNTIF({column_name} IS NOT NULL) AS non_null_count,
        SAFE_DIVIDE(
            COUNT(DISTINCT {column_name}),
            COUNTIF({column_name} IS NOT NULL)
        ) AS cardinality_ratio
    FROM
        `{table_name}`
),

top_values AS (
    SELECT
        CAST({column_name} AS STRING) AS value,
        COUNT(*) AS frequency,
        SAFE_DIVIDE(COUNT(*), SUM(COUNT(*)) OVER ()) AS relative_frequency
    FROM
        `{table_name}`
    WHERE
        {column_name} IS NOT NULL
    GROUP BY
        value
    ORDER BY
        frequency DESC
    LIMIT 20
)

SELECT
    c.column_name,
    c.distinct_count,
    c.non_null_count,
    c.cardinality_ratio,
    t.value,
    t.frequency,
    t.relative_frequency
FROM
    cardinality c
CROSS JOIN
    top_values t
ORDER BY
    t.frequency DESC
