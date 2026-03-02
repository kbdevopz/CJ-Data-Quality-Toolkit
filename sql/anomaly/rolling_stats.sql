-- Rolling-window statistics for BigQuery.
--
-- Computes a trailing rolling mean and population standard deviation over a
-- configurable window size using ROWS BETWEEN ... PRECEDING AND CURRENT ROW.
--
-- Parameters (replace before executing):
--   {TABLE}       -- fully-qualified table, e.g. `project.dataset.table`
--   {DATE_COL}    -- name of the date column
--   {VALUE_COL}   -- name of the numeric value column
--   {PRECEDING}   -- window size minus one (e.g. 11 for a 12-period window)

SELECT
    {DATE_COL},
    {VALUE_COL},
    AVG({VALUE_COL}) OVER (
        ORDER BY {DATE_COL}
        ROWS BETWEEN {PRECEDING} PRECEDING AND CURRENT ROW
    ) AS rolling_mean,
    STDDEV_POP({VALUE_COL}) OVER (
        ORDER BY {DATE_COL}
        ROWS BETWEEN {PRECEDING} PRECEDING AND CURRENT ROW
    ) AS rolling_std,
    COUNT({VALUE_COL}) OVER (
        ORDER BY {DATE_COL}
        ROWS BETWEEN {PRECEDING} PRECEDING AND CURRENT ROW
    ) AS window_count
FROM `{TABLE}`
ORDER BY {DATE_COL};
