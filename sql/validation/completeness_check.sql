-- Column Completeness Check
-- Computes null rate for each specified column.
-- Parameters: {TABLE}, {COLUMN}
--
-- Run once per column or use UNION ALL for batch assessment.

SELECT
  '{COLUMN}' AS column_name,
  COUNT(*) AS total_count,
  COUNTIF({COLUMN} IS NULL) AS null_count,
  SAFE_DIVIDE(COUNTIF({COLUMN} IS NULL), COUNT(*)) AS null_rate
FROM `{TABLE}`
