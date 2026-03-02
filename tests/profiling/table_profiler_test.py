"""Tests for cj_data_quality.profiling.table_profiler."""

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.profiling.table_profiler import profile_table


class TestProfileTable:
    """Tests for profile_table()."""

    def test_basic_table_profile(self) -> None:
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [90.5, 85.0, 92.3, 88.1, 95.0],
        })
        profile = profile_table(df, "students")
        assert profile.table_name == "students"
        assert profile.row_count == 5
        assert profile.column_count == 3
        assert len(profile.column_profiles) == 3

    def test_column_names_match(self) -> None:
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        profile = profile_table(df, "test")
        col_names = [cp.column_name for cp in profile.column_profiles]
        assert col_names == ["col_a", "col_b"]

    def test_duplicate_detection_no_duplicates(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        profile = profile_table(df, "unique_data")
        assert profile.duplicate_row_count == 0
        assert profile.duplicate_rate == pytest.approx(0.0)

    def test_duplicate_detection_with_duplicates(self) -> None:
        df = pd.DataFrame({
            "x": [1, 2, 3, 1, 2],
            "y": ["a", "b", "c", "a", "b"],
        })
        profile = profile_table(df, "dup_data")
        assert profile.duplicate_row_count == 2
        assert profile.duplicate_rate == pytest.approx(2 / 5)

    def test_overall_null_rate_zero(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        profile = profile_table(df, "complete")
        assert profile.overall_null_rate == pytest.approx(0.0)

    def test_overall_null_rate_with_nulls(self) -> None:
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [np.nan, np.nan, 6],
        })
        # 3 nulls out of 6 total cells = 0.5
        profile = profile_table(df, "sparse")
        assert profile.overall_null_rate == pytest.approx(0.5)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        profile = profile_table(df, "empty")
        assert profile.row_count == 0
        assert profile.column_count == 1
        assert profile.overall_null_rate == pytest.approx(0.0)
        assert profile.duplicate_rate == pytest.approx(0.0)

    def test_clean_incarceration_profile(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        profile = profile_table(clean_incarceration_df, "incarceration")
        assert profile.row_count == 100
        assert profile.column_count == 9
        assert profile.duplicate_row_count >= 0
        # Clean data should have zero null rate
        assert profile.overall_null_rate == pytest.approx(0.0)

    def test_dirty_incarceration_profile(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        profile = profile_table(dirty_incarceration_df, "dirty_incarceration")
        # Dirty df has 200 + 10 duplicated rows = 210 rows
        assert profile.row_count == 210
        # Should detect duplicates
        assert profile.duplicate_row_count > 0
        assert profile.duplicate_rate > 0.0
        # Should have non-zero overall null rate (admission_date and race have nulls)
        assert profile.overall_null_rate > 0.0

    def test_all_columns_profiled(self) -> None:
        df = pd.DataFrame({
            "person_id": ["P001", "P002"],
            "admission_date": pd.to_datetime(["2020-01-01", "2020-06-15"]),
            "age": [25, 30],
            "race": ["WHITE", "BLACK"],
        })
        profile = profile_table(df, "mixed")
        assert len(profile.column_profiles) == 4
        # Check that each column profile has the right type
        profile_map = {cp.column_name: cp for cp in profile.column_profiles}
        assert profile_map["age"].numeric_stats is not None
        assert profile_map["admission_date"].temporal_stats is not None
