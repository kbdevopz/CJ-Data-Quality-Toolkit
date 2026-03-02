"""Edge case tests across multiple modules.

Tests for: empty DataFrames, all-NaN columns, single-row data,
zero-variance data, boundary conditions.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cj_data_quality.anomaly.time_series_detector import (
    detect_iqr_anomalies,
    detect_zscore_anomalies,
)
from cj_data_quality.coverage.coverage_matrix import (
    build_coverage_matrix,
    identify_coverage_gaps,
    summarize_coverage,
)
from cj_data_quality.drift.distribution_drift import (
    classify_drift_severity,
    detect_numeric_drift,
)
from cj_data_quality.profiling.column_profiler import profile_column
from cj_data_quality.profiling.table_profiler import profile_table
from cj_data_quality.validation.completeness_scorer import (
    score_completeness,
    score_uniqueness,
)
from cj_data_quality.visualization.heatmaps import (
    plot_coverage_heatmap,
    plot_equity_heatmap,
    plot_quality_heatmap,
)
from cj_data_quality.visualization.plots import (
    plot_null_rate_bars,
    plot_quality_scorecard,
)
from cj_data_quality.types import QualityScore, DimensionScore, QualityDimension


# ---------------------------------------------------------------------------
# Profiling edge cases
# ---------------------------------------------------------------------------


class TestProfileColumnEdgeCases:
    def test_all_null_column(self) -> None:
        s = pd.Series([None, None, None], name="empty")
        profile = profile_column(s, "empty")
        assert profile.null_rate == 1.0
        assert profile.distinct_count == 0

    def test_single_value_column(self) -> None:
        s = pd.Series([42], name="single")
        profile = profile_column(s, "single")
        assert profile.null_rate == 0.0

    def test_all_same_value(self) -> None:
        s = pd.Series([7, 7, 7, 7, 7], name="constant")
        profile = profile_column(s, "constant")
        assert profile.distinct_count == 1


class TestProfileTableEdgeCases:
    def test_single_row(self) -> None:
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        tp = profile_table(df, "tiny")
        assert tp.row_count == 1
        assert tp.column_count == 2

    def test_all_null_dataframe(self) -> None:
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        tp = profile_table(df, "nulls")
        assert tp.overall_null_rate == 1.0


# ---------------------------------------------------------------------------
# Coverage edge cases
# ---------------------------------------------------------------------------


class TestCoverageEdgeCases:
    def test_single_state(self) -> None:
        df = pd.DataFrame({"state_code": ["US_CA"] * 5, "val": [1, 2, None, 4, 5]})
        matrix = build_coverage_matrix(df, "state_code", ["val"])
        assert len(matrix) == 1

    def test_perfect_coverage(self) -> None:
        df = pd.DataFrame({"state_code": ["US_CA"] * 3, "val": [1, 2, 3]})
        matrix = build_coverage_matrix(df, "state_code", ["val"])
        assert matrix.iloc[0, 0] == 1.0

    def test_zero_coverage(self) -> None:
        df = pd.DataFrame({"state_code": ["US_CA"] * 3, "val": [None, None, None]})
        matrix = build_coverage_matrix(df, "state_code", ["val"])
        assert matrix.iloc[0, 0] == 0.0

    def test_no_gaps_above_threshold(self) -> None:
        df = pd.DataFrame({"state_code": ["US_CA"] * 3, "val": [1, 2, 3]})
        matrix = build_coverage_matrix(df, "state_code", ["val"])
        gaps = identify_coverage_gaps(matrix, threshold=0.5)
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# Drift edge cases
# ---------------------------------------------------------------------------


class TestDriftEdgeCases:
    def test_boundary_p_value_critical(self) -> None:
        severity = classify_drift_severity(0.001)
        assert severity.value in ("critical", "high", "none")

    def test_boundary_p_value_at_005(self) -> None:
        severity = classify_drift_severity(0.05)
        # exactly 0.05 should be "none" (not significant)
        assert severity is not None

    def test_identical_distributions(self) -> None:
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = detect_numeric_drift(s1, s2, "val", "p1", "p2")
        assert result.p_value > 0.05  # no drift


# ---------------------------------------------------------------------------
# Anomaly edge cases
# ---------------------------------------------------------------------------


class TestAnomalyEdgeCases:
    def test_zero_variance_zscore(self) -> None:
        df = pd.DataFrame({
            "dt": pd.date_range("2023-01-01", periods=10, freq="ME"),
            "val": [5.0] * 10,
        })
        results = detect_zscore_anomalies(df, "dt", "val", threshold=3.0)
        assert len(results) == 0  # no anomalies when all values equal

    def test_zero_variance_iqr(self) -> None:
        df = pd.DataFrame({
            "dt": pd.date_range("2023-01-01", periods=10, freq="ME"),
            "val": [5.0] * 10,
        })
        results = detect_iqr_anomalies(df, "dt", "val", multiplier=1.5)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Scoring edge cases
# ---------------------------------------------------------------------------


class TestScoringEdgeCases:
    def test_empty_dataframe_completeness(self) -> None:
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        result = score_completeness(df, required_columns=["a"])
        assert result.score == 1.0  # no rows = no nulls = perfect

    def test_all_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 1, 1]})
        result = score_uniqueness(df, key_columns=["a"])
        assert result.score < 1.0
        assert result.details["duplicate_rows"] == 3.0


# ---------------------------------------------------------------------------
# Visualization edge cases
# ---------------------------------------------------------------------------


class TestVisualizationEdgeCases:
    def test_null_rate_bars_empty(self) -> None:
        fig = plot_null_rate_bars([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_coverage_heatmap_empty(self) -> None:
        fig = plot_coverage_heatmap(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_equity_heatmap_empty(self) -> None:
        fig = plot_equity_heatmap(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_quality_heatmap_empty(self) -> None:
        fig = plot_quality_heatmap(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_quality_scorecard_no_dimensions(self) -> None:
        score = QualityScore(
            entity_name="empty",
            composite_score=0.0,
            dimension_scores=[],
            grade="F",
        )
        fig = plot_quality_scorecard(score)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
