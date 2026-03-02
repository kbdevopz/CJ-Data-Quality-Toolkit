-- Demographic Completeness by State
-- For a given demographic column, computes per-state completeness,
-- distinct value count, and most common value with its rate.
-- Parameters: {TABLE}, {STATE_COL}, {DEMO_COL}
--
-- BigQuery dialect.

WITH base AS (
    SELECT
        {STATE_COL},
        {DEMO_COL},
        COUNT(*) OVER (PARTITION BY {STATE_COL}) AS total_rows,
        COUNT({DEMO_COL}) OVER (PARTITION BY {STATE_COL}) AS non_null_count
    FROM
        `{TABLE}`
),
completeness AS (
    SELECT
        {STATE_COL},
        ANY_VALUE(total_rows) AS total_rows,
        ANY_VALUE(non_null_count) AS non_null_count,
        SAFE_DIVIDE(ANY_VALUE(non_null_count), ANY_VALUE(total_rows)) AS completeness,
        COUNT(DISTINCT {DEMO_COL}) AS distinct_values
    FROM
        base
    GROUP BY
        {STATE_COL}
),
most_common AS (
    SELECT
        {STATE_COL},
        {DEMO_COL} AS most_common_value,
        COUNT(*) AS value_count,
        ROW_NUMBER() OVER (PARTITION BY {STATE_COL} ORDER BY COUNT(*) DESC) AS rn
    FROM
        `{TABLE}`
    WHERE
        {DEMO_COL} IS NOT NULL
    GROUP BY
        {STATE_COL}, {DEMO_COL}
)
SELECT
    c.{STATE_COL},
    c.total_rows,
    c.non_null_count,
    c.completeness,
    c.distinct_values,
    m.most_common_value,
    SAFE_DIVIDE(m.value_count, c.non_null_count) AS most_common_rate
FROM
    completeness c
LEFT JOIN
    most_common m
ON
    c.{STATE_COL} = m.{STATE_COL} AND m.rn = 1
ORDER BY
    c.{STATE_COL}
