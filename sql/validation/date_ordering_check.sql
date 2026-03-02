-- Date Ordering Validation
-- Checks that date fields follow expected temporal ordering.
-- Parameters: {TABLE}, {EARLIER_COL}, {LATER_COL}
--
-- CJ domain rule: offense_date < sentence_date < admission_date < release_date

SELECT
  '{EARLIER_COL}' AS earlier_field,
  '{LATER_COL}' AS later_field,
  COUNT(*) AS total_checked,
  COUNTIF({EARLIER_COL} > {LATER_COL}) AS violation_count,
  SAFE_DIVIDE(
    COUNTIF({EARLIER_COL} > {LATER_COL}),
    COUNT(*)
  ) AS violation_rate
FROM `{TABLE}`
WHERE {EARLIER_COL} IS NOT NULL
  AND {LATER_COL} IS NOT NULL
