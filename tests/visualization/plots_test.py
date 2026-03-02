"""Tests for visualization functions.

Tests verify that plots are created without errors and return Figure objects.
Visual correctness is verified manually via notebooks.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from cj_data_quality.types import (
    ColumnDataType,
    ColumnProfile,
    DimensionScore,
    QualityDimension,
    QualityScore,
    TableProfile,
)
from cj_data_quality.visualization.heatmaps import (
    plot_coverage_heatmap,
    plot_equity_heatmap,
    plot_quality_heatmap,
)
from cj_data_quality.visualization.plots import (
    plot_anomaly_scatter,
    plot_drift_timeline,
    plot_null_rate_bars,
    plot_profile_summary,
    plot_quality_scorecard,
)


@pytest.fixture
def sample_profiles() -> list[ColumnProfile]:
    return [
        ColumnProfile(
            column_name="person_id",
            inferred_type=ColumnDataType.IDENTIFIER,
            total_count=100,
            null_count=0,
            null_rate=0.0,
            distinct_count=100,
            cardinality_ratio=1.0,
        ),
        ColumnProfile(
            column_name="race",
            inferred_type=ColumnDataType.CATEGORICAL,
            total_count=100,
            null_count=15,
            null_rate=0.15,
            distinct_count=5,
            cardinality_ratio=0.05,
        ),
        ColumnProfile(
            column_name="admission_date",
            inferred_type=ColumnDataType.TEMPORAL,
            total_count=100,
            null_count=55,
            null_rate=0.55,
            distinct_count=80,
            cardinality_ratio=0.8,
        ),
    ]


@pytest.fixture
def sample_table_profile(sample_profiles: list[ColumnProfile]) -> TableProfile:
    return TableProfile(
        table_name="test_table",
        row_count=100,
        column_count=3,
        column_profiles=sample_profiles,
        duplicate_row_count=5,
        duplicate_rate=0.05,
        overall_null_rate=0.23,
    )


@pytest.fixture
def sample_quality_score() -> QualityScore:
    return QualityScore(
        entity_name="Test Dataset",
        composite_score=0.78,
        dimension_scores=[
            DimensionScore(
                dimension=QualityDimension.COMPLETENESS, score=0.85, weight=0.30
            ),
            DimensionScore(
                dimension=QualityDimension.CONSISTENCY, score=0.90, weight=0.25
            ),
            DimensionScore(
                dimension=QualityDimension.TIMELINESS, score=0.60, weight=0.20
            ),
            DimensionScore(
                dimension=QualityDimension.VALIDITY, score=0.70, weight=0.15
            ),
            DimensionScore(
                dimension=QualityDimension.UNIQUENESS, score=0.75, weight=0.10
            ),
        ],
        grade="C",
    )


class TestNullRateBars:
    def test_returns_figure(self, sample_profiles: list[ColumnProfile]) -> None:
        fig = plot_null_rate_bars(sample_profiles)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, sample_profiles: list[ColumnProfile]) -> None:
        fig = plot_null_rate_bars(sample_profiles, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestProfileSummary:
    def test_returns_figure(self, sample_table_profile: TableProfile) -> None:
        fig = plot_profile_summary(sample_table_profile)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestDriftTimeline:
    def test_returns_figure(self) -> None:
        drift_df = pd.DataFrame(
            {
                "period_pair": ["Q1-Q2", "Q2-Q3", "Q3-Q4"],
                "p_value": [0.8, 0.03, 0.0005],
            }
        )
        fig = plot_drift_timeline(drift_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAnomalyScatter:
    def test_returns_figure(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="MS"),
                "value": range(50),
            }
        )
        fig = plot_anomaly_scatter(df, "date", "value", [10, 30])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_anomalies(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=20, freq="MS"),
                "value": range(20),
            }
        )
        fig = plot_anomaly_scatter(df, "date", "value", [])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestQualityScorecard:
    def test_returns_figure(self, sample_quality_score: QualityScore) -> None:
        fig = plot_quality_scorecard(sample_quality_score)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCoverageHeatmap:
    def test_returns_figure(self) -> None:
        matrix = pd.DataFrame(
            {"metric_a": [0.95, 0.80, 0.30], "metric_b": [1.0, 0.60, 0.90]},
            index=["US_CA", "US_TX", "US_NY"],
        )
        fig = plot_coverage_heatmap(matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEquityHeatmap:
    def test_returns_figure(self) -> None:
        eq_df = pd.DataFrame(
            {
                "state_code": ["US_CA", "US_CA", "US_TX", "US_TX"],
                "field_name": ["race", "sex", "race", "sex"],
                "completeness": [0.95, 1.0, 0.60, 0.98],
            }
        )
        fig = plot_equity_heatmap(eq_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestQualityHeatmap:
    def test_returns_figure(self) -> None:
        q_df = pd.DataFrame(
            {
                "entity": ["State A", "State A", "State B", "State B"],
                "dimension": ["completeness", "consistency", "completeness", "consistency"],
                "score": [0.9, 0.85, 0.6, 0.7],
            }
        )
        fig = plot_quality_heatmap(q_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
