-- Cross-State Coverage Matrix
-- Computes data completeness (1 - null_rate) for each state-metric pair.
-- Parameters: {TABLE}, {STATE_COL}, and metric columns as needed.
--
-- BigQuery dialect.

SELECT
    {STATE_COL},
    COUNT(*) AS total_rows,
    1.0 - COUNTIF(total_population IS NULL) / COUNT(*) AS total_population_completeness,
    1.0 - COUNTIF(admission_count IS NULL) / COUNT(*) AS admission_count_completeness,
    1.0 - COUNTIF(release_count IS NULL) / COUNT(*) AS release_count_completeness,
    1.0 - COUNTIF(incarceration_population IS NULL) / COUNT(*) AS incarceration_population_completeness,
    1.0 - COUNTIF(supervision_population IS NULL) / COUNT(*) AS supervision_population_completeness,
    1.0 - COUNTIF(parole_population IS NULL) / COUNT(*) AS parole_population_completeness,
    1.0 - COUNTIF(probation_population IS NULL) / COUNT(*) AS probation_population_completeness,
    1.0 - COUNTIF(revocation_count IS NULL) / COUNT(*) AS revocation_count_completeness
FROM
    `{TABLE}`
GROUP BY
    {STATE_COL}
ORDER BY
    {STATE_COL}
