# Criminal Justice Data Quality Profiling & Validation Toolkit

A comprehensive data quality toolkit purpose-built for criminal justice data, designed to complement [Recidiviz's](https://www.recidiviz.org/) existing validation framework. Built with their exact tech stack (Python 3.11, pandas, attrs, BigQuery SQL) and coding conventions.

## Why This Exists

Recidiviz's validation framework (`recidiviz/validation/`) provides two check types: **existence** and **sameness**. These are essential but leave significant gaps in data quality monitoring. This toolkit fills those gaps:

| Capability | Recidiviz Today | This Toolkit |
|---|---|---|
| Existence checks | Yes | Yes |
| Sameness checks | Yes | Yes |
| Column profiling (null rates, cardinality, distributions) | No | **Yes** |
| Semantic type inference | No | **Yes** |
| Cross-state coverage matrices | No | **Yes** |
| Demographic equity coverage | No | **Yes** |
| Distribution drift detection (KS test, chi-squared) | No | **Yes** |
| Temporal drift analysis | No | **Yes** |
| Anomaly detection (Z-score, IQR, rolling window) | No | **Yes** |
| Population spike detection | No | **Yes** |
| Temporal consistency validation | No | **Yes** |
| Referential integrity checks | No | **Yes** |
| Composite quality scoring (0-1.0) | No | **Yes** |
| BigQuery SQL generators | No | **Yes** |
| Recidiviz-styled visualizations | No | **Yes** |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the interactive dashboard
streamlit run app.py

# Generate sample data
python data/download_bjs_data.py

# Run tests
pytest tests/ -v --cov=cj_data_quality

# Open notebooks
jupyter notebook notebooks/
```

## Python API

```python
import pandas as pd
from cj_data_quality import profile_table, build_coverage_matrix, compute_composite_score

df = pd.read_csv("your_corrections_data.csv")

# Profile a table
profile = profile_table(df, "corrections_data")
print(f"Null rate: {profile.overall_null_rate:.1%}")

# Build a cross-state coverage matrix
matrix = build_coverage_matrix(df, "state_code", ["admission_count", "release_count"])

# Score overall data quality (5 dimensions, letter grade)
score = compute_composite_score(df, "corrections_data")
print(f"Grade: {score.grade} ({score.composite_score:.2f})")
```

## SQL Generators

Every Python analysis has a BigQuery SQL equivalent:

```python
from cj_data_quality.validation.sql_generators import generate_date_ordering_sql

sql = generate_date_ordering_sql(
    "project.dataset.corrections",
    earlier_col="offense_date",
    later_col="admission_date",
)
print(sql)  # Ready-to-run BigQuery SQL
```

## Project Structure

```
app.py                    # Streamlit interactive dashboard (6 tabs)
cj_data_quality/          # Main package
  profiling/              # Column & table profiling, type inference
  drift/                  # KS test, chi-squared, temporal drift
  anomaly/                # Z-score, IQR, rolling window, spike detection
  coverage/               # State x metric matrices, equity coverage
  validation/             # Date checks, referential integrity, scoring
  visualization/          # Recidiviz-styled charts & heatmaps

notebooks/                # Interactive analysis notebooks
  01_data_profiling.ipynb
  02_cross_state_coverage.ipynb
  03_temporal_drift_analysis.ipynb
  04_anomaly_detection.ipynb
  05_equity_data_quality.ipynb

sql/                      # BigQuery-compatible SQL for warehouse-scale
tests/                    # pytest suite with >85% coverage
data/                     # Synthetic data generation
```

## Notebooks

1. **Data Profiling** — Column-level statistics, type inference, null rate visualization
2. **Cross-State Coverage** — State x metric completeness heatmap, gap identification
3. **Temporal Drift Analysis** — KS/chi-squared drift detection across quarters
4. **Anomaly Detection** — Z-score/IQR spike detection in population counts
5. **Equity Data Quality** — Demographic completeness analysis + composite scoring

## Design Decisions

- **Frozen attrs classes** — All data types are immutable, matching Recidiviz's codebase conventions
- **SQL generators alongside pandas** — Every analysis includes a BigQuery-compatible SQL equivalent, demonstrating warehouse-scale thinking
- **Synthetic data with known issues** — Controllable, reproducible demonstrations where the toolkit catches deliberately injected quality problems
- **Recidiviz visual identity** — Same 11-color palette, same plot conventions
- **scipy for statistical tests** — KS test and chi-squared are standard; value is knowing when to apply them to CJ data
- **Type hints everywhere** — Matches their `disallow_untyped_defs = true` mypy configuration

## Criminal Justice Domain Knowledge

This toolkit understands CJ-specific data patterns:
- **Date ordering rules**: offense < sentence < admission < release
- **Demographic completeness**: Race, ethnicity, sex fields with equity analysis
- **Population metrics**: Incarceration, supervision, parole, probation counts
- **State reporting patterns**: Quarterly cadence, varying completeness by state
- **Common quality issues**: Missing demographics, date inversions, reporting gaps, population spikes

## Tech Stack

- Python 3.11
- pandas, numpy — Data manipulation
- attrs — Frozen data classes (Recidiviz convention)
- scipy — Statistical tests (KS, chi-squared)
- matplotlib, seaborn — Visualization
- Streamlit — Interactive dashboard
- BigQuery SQL — Warehouse-scale queries
- pytest — Testing with >85% coverage target

## License

Apache 2.0 — see [LICENSE](LICENSE).
