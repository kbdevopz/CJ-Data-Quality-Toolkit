"""Tests for cj_data_quality.profiling.column_profiler."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.profiling.column_profiler import (
    compute_numeric_stats,
    compute_temporal_stats,
    profile_column,
)
from cj_data_quality.types import ColumnDataType


class TestProfileColumn:
    """Tests for profile_column()."""

    def test_null_rate_all_present(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5], name="val")
        profile = profile_column(series, "val")
        assert profile.null_rate == 0.0
        assert profile.null_count == 0
        assert profile.total_count == 5

    def test_null_rate_with_nulls(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="val")
        profile = profile_column(series, "val")
        assert profile.null_count == 2
        assert profile.null_rate == pytest.approx(0.4)

    def test_null_rate_all_null(self) -> None:
        series = pd.Series([np.nan, np.nan, np.nan], name="val")
        profile = profile_column(series, "val")
        assert profile.null_rate == pytest.approx(1.0)
        assert profile.distinct_count == 0

    def test_top_k_values(self) -> None:
        series = pd.Series(["a", "b", "a", "c", "a", "b", "d"])
        profile = profile_column(series, "letters")
        # "a" appears 3 times, "b" 2 times
        top_val_dict = dict(profile.top_values)
        assert top_val_dict["a"] == 3
        assert top_val_dict["b"] == 2

    def test_top_k_respects_limit(self) -> None:
        # Create data with 15 distinct values
        values = [f"v{i}" for i in range(15)] * 2
        series = pd.Series(values)
        profile = profile_column(series, "many_vals", top_k=5)
        assert len(profile.top_values) == 5

    def test_cardinality_ratio(self) -> None:
        # All unique values -> ratio = 1.0
        series = pd.Series(["a", "b", "c", "d", "e"])
        profile = profile_column(series, "unique_col")
        assert profile.cardinality_ratio == pytest.approx(1.0)

    def test_cardinality_ratio_repeated(self) -> None:
        series = pd.Series(["a", "a", "a", "a", "b"])
        profile = profile_column(series, "low_card")
        # 2 distinct / 5 non-null = 0.4
        assert profile.cardinality_ratio == pytest.approx(0.4)

    def test_numeric_column_produces_numeric_stats(self) -> None:
        series = pd.Series([10, 20, 30, 40, 50])
        profile = profile_column(series, "numeric_col")
        assert profile.numeric_stats is not None
        assert profile.numeric_stats.mean == pytest.approx(30.0)
        assert profile.numeric_stats.min_value == pytest.approx(10.0)
        assert profile.numeric_stats.max_value == pytest.approx(50.0)

    def test_datetime_column_produces_temporal_stats(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-06-15", "2020-12-31"])
        series = pd.Series(dates)
        profile = profile_column(series, "some_date")
        assert profile.temporal_stats is not None
        assert profile.temporal_stats.min_date == date(2020, 1, 1)
        assert profile.temporal_stats.max_date == date(2020, 12, 31)

    def test_identifier_column_type(self) -> None:
        series = pd.Series(["P001", "P002", "P003"])
        profile = profile_column(series, "person_id")
        assert profile.inferred_type == ColumnDataType.IDENTIFIER

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=float)
        profile = profile_column(series, "empty")
        assert profile.total_count == 0
        assert profile.null_rate == 0.0
        assert profile.distinct_count == 0
        assert profile.top_values == []

    def test_profile_on_clean_df(self, clean_incarceration_df: pd.DataFrame) -> None:
        profile = profile_column(
            clean_incarceration_df["age"], "age"
        )
        assert profile.total_count == 100
        assert profile.null_count == 0
        assert profile.numeric_stats is not None


class TestComputeNumericStats:
    """Tests for compute_numeric_stats()."""

    def test_known_values(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_numeric_stats(series)
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min_value == pytest.approx(1.0)
        assert stats.max_value == pytest.approx(5.0)
        assert stats.p25 == pytest.approx(2.0)
        assert stats.p75 == pytest.approx(4.0)

    def test_std_known_values(self) -> None:
        series = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        stats = compute_numeric_stats(series)
        # pandas uses sample std (ddof=1) by default, giving ~2.14
        assert stats.std == pytest.approx(2.138, abs=0.01)

    def test_skewness_symmetric(self) -> None:
        # Symmetric distribution should have skewness near 0
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = compute_numeric_stats(series)
        assert stats.skewness is not None
        assert abs(stats.skewness) < 0.5

    def test_skewness_right_skewed(self) -> None:
        # Right-skewed distribution
        series = pd.Series([1, 1, 1, 2, 2, 3, 10, 50, 100])
        stats = compute_numeric_stats(series)
        assert stats.skewness is not None
        assert stats.skewness > 0

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=float)
        stats = compute_numeric_stats(series)
        assert stats.mean == 0.0
        assert stats.skewness is None

    def test_single_value(self) -> None:
        series = pd.Series([42.0])
        stats = compute_numeric_stats(series)
        assert stats.mean == pytest.approx(42.0)
        assert stats.median == pytest.approx(42.0)
        assert stats.min_value == pytest.approx(42.0)
        assert stats.max_value == pytest.approx(42.0)


class TestComputeTemporalStats:
    """Tests for compute_temporal_stats()."""

    def test_date_range(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-01-31"])
        stats = compute_temporal_stats(pd.Series(dates))
        assert stats.min_date == date(2020, 1, 1)
        assert stats.max_date == date(2020, 1, 31)
        assert stats.date_range_days == 30

    def test_most_common_day_of_week(self) -> None:
        # Create dates that are mostly on Mondays (dayofweek=0)
        mondays = pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20"])
        other = pd.to_datetime(["2020-01-07"])
        all_dates = pd.Series(pd.DatetimeIndex(mondays.append(other)))
        stats = compute_temporal_stats(all_dates)
        assert stats.most_common_day_of_week == 0  # Monday

    def test_most_common_month(self) -> None:
        # 3 dates in January, 1 in February
        dates = pd.to_datetime([
            "2020-01-01", "2020-01-15", "2020-01-28",
            "2020-02-10",
        ])
        stats = compute_temporal_stats(pd.Series(dates))
        assert stats.most_common_month == 1  # January

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype="datetime64[ns]")
        stats = compute_temporal_stats(series)
        assert stats.date_range_days == 0
        assert stats.most_common_day_of_week is None
