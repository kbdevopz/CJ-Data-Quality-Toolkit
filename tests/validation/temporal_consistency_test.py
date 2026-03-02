"""Tests for temporal consistency validation checks."""

from __future__ import annotations

import pandas as pd
import pytest

from cj_data_quality.validation.temporal_consistency import (
    check_date_ordering,
    check_date_reasonableness,
    find_date_violations,
)


class TestCheckDateOrdering:
    """Tests for check_date_ordering."""

    def test_clean_data_has_low_violations(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should have a low ordering violation rate.

        Note: the fixture generates admission_date and release_date with
        independent random offsets, so a small number of inversions is
        expected.  We verify the check runs and returns a low rate.
        """
        result = check_date_ordering(
            clean_incarceration_df,
            date_pairs=[("admission_date", "release_date")],
        )
        assert len(result) == 1
        row = result.iloc[0]
        assert row["total_checked"] == len(clean_incarceration_df)
        # Violation rate should be small (< 10%)
        assert row["violation_rate"] < 0.10

    def test_perfectly_ordered_data_has_zero_violations(self) -> None:
        """A hand-crafted DataFrame with correct ordering has 0 violations."""
        df = pd.DataFrame(
            {
                "admission_date": pd.to_datetime(
                    ["2020-01-01", "2020-06-15", "2021-03-10"]
                ),
                "release_date": pd.to_datetime(
                    ["2020-06-01", "2021-01-01", "2022-01-01"]
                ),
            }
        )
        result = check_date_ordering(
            df, date_pairs=[("admission_date", "release_date")]
        )
        assert result.iloc[0]["violation_count"] == 0
        assert result.iloc[0]["violation_rate"] == 0.0

    def test_dirty_data_catches_inversions(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data should detect date ordering violations."""
        result = check_date_ordering(
            dirty_incarceration_df,
            date_pairs=[("admission_date", "release_date")],
        )
        assert len(result) == 1
        row = result.iloc[0]
        assert row["violation_count"] > 0
        assert row["violation_rate"] > 0.0

    def test_missing_columns_are_skipped(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Date pairs referencing absent columns should be silently skipped."""
        result = check_date_ordering(
            clean_incarceration_df,
            date_pairs=[("nonexistent_start", "nonexistent_end")],
        )
        assert len(result) == 0

    def test_defaults_to_cj_rules(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """When no date_pairs given, uses CJ_DATE_ORDERING_RULES."""
        result = check_date_ordering(clean_incarceration_df)
        # Only admission_date / release_date are present in the fixture
        assert len(result) == 1
        assert result.iloc[0]["earlier_field"] == "admission_date"
        assert result.iloc[0]["later_field"] == "release_date"

    def test_all_null_pair_returns_zero(self) -> None:
        """When both columns are all-null, total_checked should be 0."""
        df = pd.DataFrame(
            {
                "admission_date": pd.Series([pd.NaT, pd.NaT]),
                "release_date": pd.Series([pd.NaT, pd.NaT]),
            }
        )
        result = check_date_ordering(
            df, date_pairs=[("admission_date", "release_date")]
        )
        assert result.iloc[0]["total_checked"] == 0
        assert result.iloc[0]["violation_count"] == 0


class TestCheckDateReasonableness:
    """Tests for check_date_reasonableness."""

    def test_clean_data_has_no_unreasonable_dates(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should have no unreasonable dates."""
        result = check_date_reasonableness(clean_incarceration_df)
        for _, row in result.iterrows():
            assert row["before_earliest_count"] == 0
            assert row["after_latest_count"] == 0
            assert row["unreasonable_rate"] == 0.0

    def test_dirty_data_catches_unreasonable_dates(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data has dates before 1900 and future dates."""
        result = check_date_reasonableness(dirty_incarceration_df)
        # At least one column should have unreasonable dates
        total_before = result["before_earliest_count"].sum()
        total_after = result["after_latest_count"].sum()
        assert total_before > 0
        assert total_after > 0

    def test_custom_earliest_boundary(self) -> None:
        """Custom earliest boundary should flag dates below it."""
        df = pd.DataFrame(
            {
                "event_date": pd.to_datetime(
                    ["2020-01-01", "1950-06-15", "2000-12-31"]
                ),
            }
        )
        result = check_date_reasonableness(
            df, date_columns=["event_date"], earliest="2000-01-01"
        )
        assert result.iloc[0]["before_earliest_count"] == 1

    def test_custom_latest_boundary(self) -> None:
        """Custom latest boundary should flag dates above it."""
        df = pd.DataFrame(
            {
                "event_date": pd.to_datetime(
                    ["2020-01-01", "2025-06-15", "2030-12-31"]
                ),
            }
        )
        result = check_date_reasonableness(
            df, date_columns=["event_date"], latest="2026-01-01"
        )
        assert result.iloc[0]["after_latest_count"] == 1


class TestFindDateViolations:
    """Tests for find_date_violations."""

    def test_returns_violating_rows(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Should return the actual rows where admission > release."""
        violations = find_date_violations(
            dirty_incarceration_df, "admission_date", "release_date"
        )
        assert len(violations) > 0
        # Every returned row should have admission > release
        for _, row in violations.iterrows():
            assert row["admission_date"] > row["release_date"]

    def test_perfectly_ordered_data_returns_empty(self) -> None:
        """A hand-crafted correctly-ordered DataFrame returns no violations."""
        df = pd.DataFrame(
            {
                "admission_date": pd.to_datetime(
                    ["2020-01-01", "2020-06-15", "2021-03-10"]
                ),
                "release_date": pd.to_datetime(
                    ["2020-06-01", "2021-01-01", "2022-01-01"]
                ),
            }
        )
        violations = find_date_violations(df, "admission_date", "release_date")
        assert len(violations) == 0
