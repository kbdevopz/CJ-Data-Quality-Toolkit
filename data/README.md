# Data Directory

## Synthetic Corrections Data

The file `sample/corrections_data.csv` contains **synthetic** data generated
to mimic the structure of Bureau of Justice Statistics (BJS) National
Corrections Reporting Program data.  No real individual records are included.

### Why synthetic?

Real BJS data requires a manual download from <https://bjs.ojp.gov/> and is
subject to usage restrictions.  The synthetic dataset allows the CJ Data
Quality toolkit to be demonstrated and tested without any external
dependencies.

### Intentionally injected quality issues

The generator deliberately introduces the following problems so that the
toolkit's profiling, validation, and anomaly-detection modules have realistic
issues to surface:

| Issue | Description |
|---|---|
| Varying null rates | Some states report 5% nulls; others exceed 40%. |
| Date inversions | ~2% of rows have `release_date` before `admission_date`. |
| Population spikes | Random state-quarters contain population counts 5-10x normal. |
| Missing reporting periods | Low-quality states are missing some quarterly reports. |
| Missing demographics | `race` and `ethnicity` null rates vary widely by state. |
| Ancient dates | A handful of dates are set before 1900 (clearly invalid). |
| Future dates | A handful of dates extend past 2025. |

### How to regenerate

From the project root:

```bash
python data/download_bjs_data.py
```

This writes `data/sample/corrections_data.csv` (approximately 50,000 rows).
The generator uses a fixed random seed (`42`) for reproducibility.

You can also generate and load programmatically:

```python
from cj_data_quality.sample_data import generate_and_load

df = generate_and_load(n_records=50000, seed=42)
```

### Columns

| Column | Type | Description |
|---|---|---|
| `person_id` | str | Unique synthetic person identifier |
| `state_code` | str | Recidiviz-style state code (e.g., `US_CA`) |
| `admission_date` | date | Date of admission to custody |
| `release_date` | date | Date of release from custody |
| `offense_date` | date | Date the offense occurred |
| `sentence_date` | date | Date the sentence was imposed |
| `facility_id` | str | Synthetic facility identifier |
| `race` | str | Race category |
| `ethnicity` | str | Ethnicity category |
| `sex` | str | Sex category |
| `age` | int | Age at admission |
| `age_group` | str | Categorical age group |
| `total_population` | int | Facility/state population count |
| `admission_count` | int | Number of admissions in the period |
| `release_count` | int | Number of releases in the period |
| `reporting_date` | date | First day of the reporting quarter |
