# Changelog

All notable changes to the CJ Data Quality Toolkit are documented here.

## [0.1.0] - 2026-03-02

### Added
- **Profiling**: Column and table profilers with type inference, null rate, distinct count, and numeric/temporal statistics.
- **Coverage**: Cross-state coverage matrix builder, gap identification, and demographic equity analysis.
- **Drift Detection**: Numeric (KS test), categorical (chi-squared), and temporal drift detectors with severity classification.
- **Anomaly Detection**: Z-score, IQR, and rolling-window time-series anomaly detectors; spike detector; cross-state population anomaly detection.
- **Validation**: Completeness scorer, uniqueness scorer, date ordering checks, and composite quality scoring with letter grades.
- **Visualization**: Null-rate bar charts, profile summary dashboards, drift timelines, anomaly scatter plots, coverage/equity/quality heatmaps, and quality scorecards — all styled with the Recidiviz palette.
- **SQL Generators**: BigQuery-dialect SQL generators for anomaly detection, coverage analysis, drift detection, and validation queries.
- **Streamlit Dashboard** (`app.py`): Interactive 6-tab dashboard with Overview & Profiling, Coverage Matrix, Demographic Equity, Drift Detection, Anomaly Detection, and Quality Scoring tabs.
- **Sample Data**: Synthetic criminal justice dataset generator with realistic state-level patterns, demographic distributions, and temporal trends.
- **CI/CD**: GitHub Actions workflow with Python 3.11/3.12 matrix, pytest with 85% coverage threshold, and mypy type checking.
- **277 tests** across all modules with edge case coverage.
- Apache 2.0 license.
