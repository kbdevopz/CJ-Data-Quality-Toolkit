"""Criminal Justice Data Quality Profiling & Validation Toolkit."""

from cj_data_quality.profiling import profile_column, profile_table
from cj_data_quality.drift import detect_numeric_drift, detect_categorical_drift, detect_temporal_drift
from cj_data_quality.anomaly import (
    detect_iqr_anomalies,
    detect_missing_periods,
    detect_population_anomalies,
    detect_rolling_anomalies,
    detect_spikes,
    detect_zscore_anomalies,
)
from cj_data_quality.coverage import build_coverage_matrix, analyze_demographic_completeness
from cj_data_quality.validation import compute_composite_score, check_date_ordering
from cj_data_quality.visualization import (
    plot_coverage_heatmap,
    plot_equity_heatmap,
    plot_quality_scorecard,
    plot_null_rate_bars,
)

__all__ = [
    "analyze_demographic_completeness",
    "build_coverage_matrix",
    "check_date_ordering",
    "compute_composite_score",
    "detect_categorical_drift",
    "detect_iqr_anomalies",
    "detect_missing_periods",
    "detect_numeric_drift",
    "detect_population_anomalies",
    "detect_rolling_anomalies",
    "detect_spikes",
    "detect_temporal_drift",
    "detect_zscore_anomalies",
    "plot_coverage_heatmap",
    "plot_equity_heatmap",
    "plot_null_rate_bars",
    "plot_quality_scorecard",
    "profile_column",
    "profile_table",
]
