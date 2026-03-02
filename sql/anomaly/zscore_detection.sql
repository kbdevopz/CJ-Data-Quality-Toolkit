-- Z-score anomaly detection for BigQuery.
--
-- Computes the population Z-score of each row in the target table and
-- returns only those rows whose absolute Z-score exceeds the threshold.
--
-- Parameters (replace before executing):
--   {TABLE}       -- fully-qualified table, e.g. `project.dataset.table`
--   {DATE_COL}    -- name of the date column
--   {VALUE_COL}   -- name of the numeric value column
--   {THRESHOLD}   -- absolute Z-score cutoff (e.g. 3.0)

WITH stats AS (
    SELECT
        AVG({VALUE_COL}) AS mean_val,
        STDDEV_POP({VALUE_COL}) AS std_val
    FROM `{TABLE}`
),
scored AS (
    SELECT
        t.{DATE_COL},
        t.{VALUE_COL},
        s.mean_val,
        s.std_val,
        SAFE_DIVIDE(t.{VALUE_COL} - s.mean_val, s.std_val) AS zscore
    FROM `{TABLE}` t
    CROSS JOIN stats s
)
SELECT
    {DATE_COL},
    {VALUE_COL},
    mean_val,
    std_val,
    zscore,
    ABS(zscore) AS abs_zscore
FROM scored
WHERE ABS(zscore) > {THRESHOLD}
ORDER BY ABS(zscore) DESC;
