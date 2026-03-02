"""Tests for temporal drift detection functions.

Covers:
- Temporal drift detection on the multi_state_population_df fixture.
- Per-group drift analysis.
- Drift summary table generation.
- Edge cases: missing columns, empty results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.drift.temporal_drift import (
    detect_temporal_drift,
    summarize_drift_over_time,
)
from cj_data_quality.types import DriftResult, DriftSeverity


# ---------------------------------------------------------------------------
# detect_temporal_drift
# ---------------------------------------------------------------------------


class TestDetectTemporalDrift:
    """Tests for temporal drift detection across consecutive periods."""

    def test_basic_quarterly_drift(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Should detect drift across consecutive quarters in population data."""
        results: list[DriftResult] = detect_temporal_drift(
            df=multi_state_population_df,
            date_col="reporting_date",
            value_col="total_population",
            period="Q",
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, DriftResult) for r in results)

        # All results should have KS test name
        assert all(r.test_name == "ks_2samp" for r in results)

        # Periods should be consecutive
        for r in results:
            assert r.reference_period != r.comparison_period

    def test_per_group_drift(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Should produce per-state drift results when group_col is provided."""
        results: list[DriftResult] = detect_temporal_drift(
            df=multi_state_population_df,
            date_col="reporting_date",
            value_col="total_population",
            period="Q",
            group_col="state_code",
        )

        assert isinstance(results, list)
        assert len(results) > 0

        # Should have results for multiple states
        column_names: set[str] = {r.column_name for r in results}
        # Column names should include state code info, e.g. "total_population[US_CA]"
        assert any("US_CA" in cn for cn in column_names)
        assert any("US_TX" in cn for cn in column_names)

    def test_ca_spike_detected_with_synthetic_data(self) -> None:
        """A large spike in one group should be detectable when enough data exists.

        The multi_state_population_df fixture has only 1 row per state per
        quarter, which is too few for the KS test to have power. Here we
        construct a synthetic dataset with many observations per period so
        that a 5x spike is reliably detected.
        """
        np.random.seed(42)
        dates_q2 = pd.date_range("2022-04-01", "2022-06-30", freq="D")
        dates_q3 = pd.date_range("2022-07-01", "2022-09-30", freq="D")

        # Q2: normal distribution around 20000
        q2_values = np.random.normal(20000, 1000, len(dates_q2))
        # Q3: spiked distribution around 100000 (5x)
        q3_values = np.random.normal(100000, 1000, len(dates_q3))

        df = pd.DataFrame(
            {
                "date": list(dates_q2) + list(dates_q3),
                "value": list(q2_values) + list(q3_values),
                "state": ["US_CA"] * (len(dates_q2) + len(dates_q3)),
            }
        )

        results: list[DriftResult] = detect_temporal_drift(
            df=df,
            date_col="date",
            value_col="value",
            period="Q",
            group_col="state",
        )

        assert len(results) > 0
        # The spike should produce CRITICAL drift
        assert any(r.severity == DriftSeverity.CRITICAL for r in results)

    def test_per_group_drift_produces_results_for_ca(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Per-group analysis should produce results for each state including CA."""
        results: list[DriftResult] = detect_temporal_drift(
            df=multi_state_population_df,
            date_col="reporting_date",
            value_col="total_population",
            period="Q",
            group_col="state_code",
        )

        ca_results: list[DriftResult] = [
            r for r in results if "US_CA" in r.column_name
        ]
        assert len(ca_results) > 0

        # With only 1 observation per state per quarter, KS test has low power,
        # so we just verify results are produced and have valid structure.
        for r in ca_results:
            assert r.test_name == "ks_2samp"
            assert 0.0 <= r.p_value <= 1.0
            assert r.statistic >= 0.0

    def test_stable_distribution_no_drift(self) -> None:
        """Data from the same distribution across periods should show no drift."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        values = np.random.normal(100, 10, len(dates))

        df = pd.DataFrame({"date": dates, "value": values})

        results: list[DriftResult] = detect_temporal_drift(
            df=df,
            date_col="date",
            value_col="value",
            period="Q",
        )

        # Most results should be NONE severity
        none_count = sum(1 for r in results if r.severity == DriftSeverity.NONE)
        assert none_count / len(results) >= 0.5, (
            "Expected majority of periods to show no drift for stable data"
        )

    def test_monthly_period(self) -> None:
        """Should work with monthly periods as well as quarterly."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
        values = np.random.normal(50, 5, len(dates))

        df = pd.DataFrame({"date": dates, "value": values})

        results: list[DriftResult] = detect_temporal_drift(
            df=df,
            date_col="date",
            value_col="value",
            period="M",
        )

        assert len(results) == 11  # 12 months -> 11 consecutive pairs

    def test_missing_column_raises(self) -> None:
        """Should raise KeyError if a required column is missing."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(KeyError, match="nonexistent"):
            detect_temporal_drift(df, "nonexistent", "b")

        with pytest.raises(KeyError, match="nonexistent"):
            detect_temporal_drift(df, "a", "nonexistent")

    def test_missing_group_column_raises(self) -> None:
        """Should raise KeyError if group_col does not exist."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=10),
                "value": range(10),
            }
        )

        with pytest.raises(KeyError, match="state_code"):
            detect_temporal_drift(df, "date", "value", group_col="state_code")

    def test_single_period_returns_empty(self) -> None:
        """A single period means no consecutive pairs, so return empty list."""
        dates = pd.date_range("2022-01-01", "2022-03-31", freq="D")
        values = np.random.normal(100, 10, len(dates))

        df = pd.DataFrame({"date": dates, "value": values})

        results: list[DriftResult] = detect_temporal_drift(
            df=df,
            date_col="date",
            value_col="value",
            period="Q",
        )

        assert results == []


# ---------------------------------------------------------------------------
# summarize_drift_over_time
# ---------------------------------------------------------------------------


class TestSummarizeDriftOverTime:
    """Tests for the drift summary pivot table function."""

    def test_summarize_returns_dataframe(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Summary should be a DataFrame with expected columns."""
        results: list[DriftResult] = detect_temporal_drift(
            df=multi_state_population_df,
            date_col="reporting_date",
            value_col="total_population",
            period="Q",
        )

        summary: pd.DataFrame = summarize_drift_over_time(results)

        assert isinstance(summary, pd.DataFrame)
        expected_cols = [
            "column_name",
            "reference_period",
            "comparison_period",
            "statistic",
            "p_value",
            "severity",
        ]
        assert list(summary.columns) == expected_cols
        assert len(summary) == len(results)

    def test_severity_values_are_strings(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Severity column in the summary should contain string values."""
        results: list[DriftResult] = detect_temporal_drift(
            df=multi_state_population_df,
            date_col="reporting_date",
            value_col="total_population",
            period="Q",
        )

        summary: pd.DataFrame = summarize_drift_over_time(results)

        valid_severities = {"none", "low", "medium", "high", "critical"}
        actual_severities = set(summary["severity"].unique())
        assert actual_severities.issubset(valid_severities)

    def test_empty_results_returns_empty_df(self) -> None:
        """Empty input list should return an empty DataFrame with correct columns."""
        summary: pd.DataFrame = summarize_drift_over_time([])

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
        assert "column_name" in summary.columns
        assert "severity" in summary.columns

    def test_summarize_with_grouped_results(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Summary should include group info from column_name field."""
        results: list[DriftResult] = detect_temporal_drift(
            df=multi_state_population_df,
            date_col="reporting_date",
            value_col="total_population",
            period="Q",
            group_col="state_code",
        )

        summary: pd.DataFrame = summarize_drift_over_time(results)

        # Column names should reflect the group
        assert any("US_CA" in cn for cn in summary["column_name"])
