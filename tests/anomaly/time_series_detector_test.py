"""Tests for cj_data_quality.anomaly.time_series_detector."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.anomaly.time_series_detector import (
    detect_iqr_anomalies,
    detect_missing_periods,
    detect_rolling_anomalies,
    detect_zscore_anomalies,
)
from cj_data_quality.types import AnomalyResult, AnomalyType


# ---------------------------------------------------------------------------
# Z-score detection
# ---------------------------------------------------------------------------


class TestDetectZscoreAnomalies:
    """Tests for ``detect_zscore_anomalies``."""

    def test_catches_injected_anomalies(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """Injected spikes/drops in the fixture should be flagged."""
        results: list[AnomalyResult] = detect_zscore_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            threshold=3.0,
            metric_name="test_metric",
        )
        assert len(results) > 0
        assert all(r.anomaly_type == AnomalyType.ZSCORE for r in results)
        assert all(r.deviation > 3.0 for r in results)
        assert all(isinstance(r.timestamp, date) for r in results)
        assert all(r.metric_name == "test_metric" for r in results)

    def test_no_anomalies_in_clean_series(self) -> None:
        """A constant series should produce no anomalies."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=24, freq="MS"),
                "value": [100.0] * 24,
            }
        )
        results: list[AnomalyResult] = detect_zscore_anomalies(
            df, date_col="date", value_col="value"
        )
        assert results == []

    def test_state_code_propagated(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """The optional state_code should appear on every result."""
        results: list[AnomalyResult] = detect_zscore_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            state_code="US_CA",
        )
        assert all(r.state_code == "US_CA" for r in results)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should return an empty list."""
        df: pd.DataFrame = pd.DataFrame(columns=["date", "value"])
        results: list[AnomalyResult] = detect_zscore_anomalies(
            df, date_col="date", value_col="value"
        )
        assert results == []

    def test_missing_column_raises_keyerror(self) -> None:
        df: pd.DataFrame = pd.DataFrame({"date": [1], "value": [2]})
        with pytest.raises(KeyError, match="bad_col"):
            detect_zscore_anomalies(df, date_col="date", value_col="bad_col")

    def test_lower_threshold_finds_more(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """A lower threshold should flag at least as many anomalies."""
        strict: list[AnomalyResult] = detect_zscore_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            threshold=3.0,
        )
        lenient: list[AnomalyResult] = detect_zscore_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            threshold=2.0,
        )
        assert len(lenient) >= len(strict)


# ---------------------------------------------------------------------------
# IQR detection
# ---------------------------------------------------------------------------


class TestDetectIqrAnomalies:
    """Tests for ``detect_iqr_anomalies``."""

    def test_catches_injected_anomalies(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """Extreme injected values should fall outside the IQR fences."""
        results: list[AnomalyResult] = detect_iqr_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            multiplier=1.5,
            metric_name="iqr_test",
        )
        assert len(results) > 0
        assert all(r.anomaly_type == AnomalyType.IQR for r in results)
        assert all(r.metric_name == "iqr_test" for r in results)

    def test_tight_multiplier_more_anomalies(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """A smaller multiplier should flag more points."""
        wide: list[AnomalyResult] = detect_iqr_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            multiplier=3.0,
        )
        tight: list[AnomalyResult] = detect_iqr_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            multiplier=1.0,
        )
        assert len(tight) >= len(wide)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should return an empty list."""
        df: pd.DataFrame = pd.DataFrame(columns=["date", "value"])
        results: list[AnomalyResult] = detect_iqr_anomalies(
            df, date_col="date", value_col="value"
        )
        assert results == []

    def test_missing_column_raises_keyerror(self) -> None:
        df: pd.DataFrame = pd.DataFrame({"date": [1], "value": [2]})
        with pytest.raises(KeyError, match="missing"):
            detect_iqr_anomalies(df, date_col="missing", value_col="value")


# ---------------------------------------------------------------------------
# Rolling window detection
# ---------------------------------------------------------------------------


class TestDetectRollingAnomalies:
    """Tests for ``detect_rolling_anomalies``."""

    def test_catches_injected_anomalies(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """Injected spikes/drops should breach the rolling band."""
        results: list[AnomalyResult] = detect_rolling_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            window=12,
            num_std=2.0,
            metric_name="rolling_test",
        )
        assert len(results) > 0
        assert all(r.anomaly_type == AnomalyType.ROLLING_WINDOW for r in results)
        assert all(r.metric_name == "rolling_test" for r in results)

    def test_too_short_series(self) -> None:
        """A series shorter than the window should return nothing."""
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
                "value": [100.0, 101.0, 99.0, 102.0, 98.0],
            }
        )
        results: list[AnomalyResult] = detect_rolling_anomalies(
            df, date_col="date", value_col="value", window=12
        )
        assert results == []

    def test_missing_column_raises_keyerror(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=20, freq="MS"),
                "value": [100.0] * 20,
            }
        )
        with pytest.raises(KeyError, match="nope"):
            detect_rolling_anomalies(df, date_col="date", value_col="nope", window=12)

    def test_narrow_band_more_anomalies(
        self, time_series_with_anomalies_df: pd.DataFrame
    ) -> None:
        """A smaller num_std should flag more rows."""
        wide: list[AnomalyResult] = detect_rolling_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            window=12,
            num_std=3.0,
        )
        narrow: list[AnomalyResult] = detect_rolling_anomalies(
            time_series_with_anomalies_df,
            date_col="date",
            value_col="value",
            window=12,
            num_std=1.5,
        )
        assert len(narrow) >= len(wide)


# ---------------------------------------------------------------------------
# Missing period detection
# ---------------------------------------------------------------------------


class TestDetectMissingPeriods:
    """Tests for ``detect_missing_periods``."""

    def test_finds_gaps(self) -> None:
        """Deliberately removed months should be detected."""
        full_dates: pd.DatetimeIndex = pd.date_range(
            "2020-01-01", "2021-12-01", freq="MS"
        )
        # Remove two months
        kept: pd.DatetimeIndex = full_dates.delete([3, 10])
        df: pd.DataFrame = pd.DataFrame(
            {"date": kept, "value": range(len(kept))}
        )

        results: list[AnomalyResult] = detect_missing_periods(
            df, date_col="date", expected_freq="MS", metric_name="gap_test"
        )
        assert len(results) == 2
        assert all(r.anomaly_type == AnomalyType.MISSING_PERIOD for r in results)
        assert all(r.observed_value == 0.0 for r in results)
        assert all(r.expected_value == 1.0 for r in results)
        assert all(r.deviation == 1.0 for r in results)
        assert all(r.threshold == 0.0 for r in results)
        assert all(r.metric_name == "gap_test" for r in results)

    def test_no_gaps_in_complete_series(self) -> None:
        """A complete monthly series should yield no missing periods."""
        dates: pd.DatetimeIndex = pd.date_range(
            "2020-01-01", "2021-12-01", freq="MS"
        )
        df: pd.DataFrame = pd.DataFrame(
            {"date": dates, "value": range(len(dates))}
        )
        results: list[AnomalyResult] = detect_missing_periods(
            df, date_col="date", expected_freq="MS"
        )
        assert results == []

    def test_quarterly_frequency(self) -> None:
        """Missing quarters should be detected when expected_freq='QS'."""
        full_dates: pd.DatetimeIndex = pd.date_range(
            "2020-01-01", "2023-12-01", freq="QS"
        )
        # Remove Q2 2021
        kept = full_dates[full_dates != pd.Timestamp("2021-04-01")]
        df: pd.DataFrame = pd.DataFrame(
            {"date": kept, "value": range(len(kept))}
        )

        results: list[AnomalyResult] = detect_missing_periods(
            df, date_col="date", expected_freq="QS"
        )
        assert len(results) == 1
        assert results[0].timestamp == date(2021, 4, 1)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should return an empty list."""
        df: pd.DataFrame = pd.DataFrame(columns=["date"])
        results: list[AnomalyResult] = detect_missing_periods(
            df, date_col="date"
        )
        assert results == []

    def test_missing_column_raises_keyerror(self) -> None:
        df: pd.DataFrame = pd.DataFrame({"date": [1]})
        with pytest.raises(KeyError, match="wrong"):
            detect_missing_periods(df, date_col="wrong")

    def test_state_code_propagated(self) -> None:
        """The state_code should be passed through to results."""
        full_dates: pd.DatetimeIndex = pd.date_range(
            "2020-01-01", "2020-06-01", freq="MS"
        )
        kept: pd.DatetimeIndex = full_dates.delete([2])
        df: pd.DataFrame = pd.DataFrame({"date": kept})

        results: list[AnomalyResult] = detect_missing_periods(
            df, date_col="date", state_code="US_TX"
        )
        assert len(results) == 1
        assert results[0].state_code == "US_TX"
