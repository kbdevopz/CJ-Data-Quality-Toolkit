-- Duplicate Detection
-- Counts duplicate rows based on a composite natural key.
-- Parameters: {TABLE}, {KEY_COLUMNS} (comma-separated list)

SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT STRUCT({KEY_COLUMNS})) AS distinct_keys,
  COUNT(*) - COUNT(DISTINCT STRUCT({KEY_COLUMNS})) AS duplicate_rows,
  SAFE_DIVIDE(
    COUNT(*) - COUNT(DISTINCT STRUCT({KEY_COLUMNS})),
    COUNT(*)
  ) AS duplicate_rate
FROM `{TABLE}`
