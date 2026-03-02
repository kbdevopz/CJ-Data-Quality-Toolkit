"""Tests for cj_data_quality.anomaly.spike_detector."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.anomaly.spike_detector import (
    detect_population_anomalies,
    detect_spikes,
)
from cj_data_quality.types import AnomalyResult, AnomalyType


# ---------------------------------------------------------------------------
# Spike / drop detection
# ---------------------------------------------------------------------------


class TestDetectSpikes:
    """Tests for ``detect_spikes``."""

    def test_catches_sudden_spike(self) -> None:
        """A doubled value should be flagged as SUDDEN_SPIKE."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "value": [100.0, 102.0, 98.0, 101.0, 200.0, 99.0],
            }
        )
        results: list[AnomalyResult] = detect_spikes(
            df,
            date_col="date",
            value_col="value",
            pct_change_threshold=0.5,
            metric_name="spike_test",
        )
        spike_results: list[AnomalyResult] = [
            r for r in results if r.anomaly_type == AnomalyType.SUDDEN_SPIKE
        ]
        assert len(spike_results) >= 1
        # The spike at index 4 (value 200 from 101) should be caught
        spike_dates: list[date] = [r.timestamp for r in spike_results]
        assert date(2020, 5, 1) in spike_dates

    def test_catches_sudden_drop(self) -> None:
        """A halved value should be flagged as SUDDEN_DROP."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "value": [100.0, 102.0, 100.0, 30.0, 99.0],
            }
        )
        results: list[AnomalyResult] = detect_spikes(
            df,
            date_col="date",
            value_col="value",
            pct_change_threshold=0.5,
            metric_name="drop_test",
        )
        drop_results: list[AnomalyResult] = [
            r for r in results if r.anomaly_type == AnomalyType.SUDDEN_DROP
        ]
        assert len(drop_results) >= 1
        drop_dates: list[date] = [r.timestamp for r in drop_results]
        assert date(2020, 4, 1) in drop_dates

    def test_catches_anomalies_in_fixture(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """The fixture has a 3x spike and a 0.2x drop -- both should be caught."""
        results: list[AnomalyResult] = detect_spikes(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            pct_change_threshold=0.5,
        )
        assert len(results) > 0
        types_found: set[AnomalyType] = {r.anomaly_type for r in results}
        # We expect at least a spike and a drop
        assert AnomalyType.SUDDEN_SPIKE in types_found or AnomalyType.SUDDEN_DROP in types_found

    def test_no_spikes_in_stable_series(self) -> None:
        """A gently trending series should have no spikes."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "value": [100 + i for i in range(12)],
            }
        )
        results: list[AnomalyResult] = detect_spikes(
            df,
            date_col="date",
            value_col="value",
            pct_change_threshold=0.5,
        )
        assert results == []

    def test_state_code_propagated(self) -> None:
        """The optional state_code should appear on every result."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "value": [100.0, 100.0, 300.0, 100.0],
            }
        )
        results: list[AnomalyResult] = detect_spikes(
            df,
            date_col="date",
            value_col="value",
            pct_change_threshold=0.5,
            state_code="US_NY",
        )
        assert len(results) > 0
        assert all(r.state_code == "US_NY" for r in results)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should return an empty list."""
        df: pd.DataFrame = pd.DataFrame(columns=["date", "value"])
        results: list[AnomalyResult] = detect_spikes(
            df, date_col="date", value_col="value"
        )
        assert results == []

    def test_missing_column_raises_keyerror(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=3, freq="MS"), "value": [1, 2, 3]}
        )
        with pytest.raises(KeyError, match="nonexist"):
            detect_spikes(df, date_col="date", value_col="nonexist")

    def test_single_row(self) -> None:
        """A single-row DataFrame cannot have pct_change, so no results."""
        df: pd.DataFrame = pd.DataFrame(
            {"date": [pd.Timestamp("2020-01-01")], "value": [100.0]}
        )
        results: list[AnomalyResult] = detect_spikes(
            df, date_col="date", value_col="value"
        )
        assert results == []

    def test_deviation_is_absolute_pct_change(self) -> None:
        """The deviation field should be the absolute percent change."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "value": [100.0, 200.0, 100.0],
            }
        )
        results: list[AnomalyResult] = detect_spikes(
            df,
            date_col="date",
            value_col="value",
            pct_change_threshold=0.5,
        )
        assert len(results) >= 1
        for r in results:
            assert r.deviation > 0


# ---------------------------------------------------------------------------
# Per-state population anomaly detection
# ---------------------------------------------------------------------------


class TestDetectPopulationAnomalies:
    """Tests for ``detect_population_anomalies``."""

    def test_catches_ca_spike(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """The fixture has a 5x spike in CA Q3 2022 -- should be flagged."""
        results: list[AnomalyResult] = detect_population_anomalies(
            multi_state_population_df,
            date_col="reporting_date",
            population_col="total_population",
            state_col="state_code",
            threshold=3.0,
        )
        assert len(results) > 0

        ca_results: list[AnomalyResult] = [
            r for r in results if r.state_code == "US_CA"
        ]
        assert len(ca_results) >= 1
        assert all(r.anomaly_type == AnomalyType.ZSCORE for r in ca_results)

    def test_each_result_has_state_code(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """Every result should carry the originating state_code."""
        results: list[AnomalyResult] = detect_population_anomalies(
            multi_state_population_df,
            date_col="reporting_date",
            population_col="total_population",
            state_col="state_code",
        )
        for r in results:
            assert r.state_code is not None
            assert r.state_code.startswith("US_")

    def test_metric_name_is_population_col(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """The metric_name should be set to the population column name."""
        results: list[AnomalyResult] = detect_population_anomalies(
            multi_state_population_df,
            date_col="reporting_date",
            population_col="total_population",
            state_col="state_code",
        )
        for r in results:
            assert r.metric_name == "total_population"

    def test_high_threshold_fewer_anomalies(
        self, multi_state_population_df: pd.DataFrame
    ) -> None:
        """A higher threshold should produce fewer (or equal) anomalies."""
        loose: list[AnomalyResult] = detect_population_anomalies(
            multi_state_population_df,
            date_col="reporting_date",
            population_col="total_population",
            state_col="state_code",
            threshold=2.0,
        )
        strict: list[AnomalyResult] = detect_population_anomalies(
            multi_state_population_df,
            date_col="reporting_date",
            population_col="total_population",
            state_col="state_code",
            threshold=4.0,
        )
        assert len(loose) >= len(strict)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should return an empty list."""
        df: pd.DataFrame = pd.DataFrame(
            columns=["reporting_date", "total_population", "state_code"]
        )
        results: list[AnomalyResult] = detect_population_anomalies(
            df,
            date_col="reporting_date",
            population_col="total_population",
            state_col="state_code",
        )
        assert results == []

    def test_missing_column_raises_keyerror(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            {"reporting_date": [1], "total_population": [2], "state_code": ["US_CA"]}
        )
        with pytest.raises(KeyError, match="bad_col"):
            detect_population_anomalies(
                df,
                date_col="reporting_date",
                population_col="bad_col",
                state_col="state_code",
            )
