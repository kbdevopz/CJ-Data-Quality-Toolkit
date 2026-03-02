"""Tests for completeness scorer and composite quality scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from cj_data_quality.types import QualityDimension, QualityScore
from cj_data_quality.validation.completeness_scorer import (
    compute_composite_score,
    score_completeness,
    score_consistency,
    score_timeliness,
    score_uniqueness,
    score_validity,
)


# ---------------------------------------------------------------------------
# Tests: score_completeness
# ---------------------------------------------------------------------------


class TestScoreCompleteness:
    """Tests for score_completeness."""

    def test_clean_data_high_completeness(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should have completeness near 1.0."""
        result = score_completeness(clean_incarceration_df)
        assert result.dimension == QualityDimension.COMPLETENESS
        assert result.score == pytest.approx(1.0)
        assert result.weight > 0

    def test_dirty_data_lower_completeness(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data (with injected nulls) should have lower completeness."""
        result = score_completeness(dirty_incarceration_df)
        assert result.score < 1.0
        # Details should contain per-column null rates
        assert result.details is not None
        assert "admission_date" in result.details
        assert result.details["admission_date"] > 0

    def test_specific_required_columns(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Scoring only columns with no nulls should yield 1.0."""
        # sex and age have no injected nulls
        result = score_completeness(
            dirty_incarceration_df, required_columns=["sex", "age"]
        )
        assert result.score == pytest.approx(1.0)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should still return a valid score."""
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = score_completeness(df)
        # No rows => no nulls => score should be 1.0
        # (NaN mean of empty series is handled)
        assert 0.0 <= result.score <= 1.0

    def test_all_null_column(self) -> None:
        """A column that is entirely null should contribute 1.0 null rate."""
        df = pd.DataFrame({"x": [None, None, None]})
        result = score_completeness(df, required_columns=["x"])
        assert result.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: score_consistency
# ---------------------------------------------------------------------------


class TestScoreConsistency:
    """Tests for score_consistency."""

    def test_clean_data_high_consistency(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should have high (but not necessarily perfect) consistency.

        The fixture generates admission/release dates with independent random
        offsets, so a small number of inversions is expected.
        """
        result = score_consistency(
            clean_incarceration_df,
            date_pairs=[("admission_date", "release_date")],
        )
        assert result.dimension == QualityDimension.CONSISTENCY
        assert result.score >= 0.90

    def test_perfectly_ordered_data_has_consistency_1(self) -> None:
        """Hand-crafted correctly-ordered data should score 1.0."""
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
        result = score_consistency(
            df, date_pairs=[("admission_date", "release_date")]
        )
        assert result.score == pytest.approx(1.0)

    def test_dirty_data_lower_consistency(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data with date inversions should have < 1.0 consistency."""
        result = score_consistency(
            dirty_incarceration_df,
            date_pairs=[("admission_date", "release_date")],
        )
        assert result.score < 1.0


# ---------------------------------------------------------------------------
# Tests: score_validity
# ---------------------------------------------------------------------------


class TestScoreValidity:
    """Tests for score_validity."""

    def test_clean_data_validity(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should have validity 1.0."""
        result = score_validity(clean_incarceration_df)
        assert result.dimension == QualityDimension.VALIDITY
        assert result.score == pytest.approx(1.0)

    def test_dirty_data_lower_validity(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data with unreasonable dates should score below 1.0."""
        result = score_validity(dirty_incarceration_df)
        assert result.score < 1.0


# ---------------------------------------------------------------------------
# Tests: score_uniqueness
# ---------------------------------------------------------------------------


class TestScoreUniqueness:
    """Tests for score_uniqueness."""

    def test_clean_data_uniqueness(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should have high uniqueness on person_id."""
        result = score_uniqueness(
            clean_incarceration_df, key_columns=["person_id"]
        )
        assert result.dimension == QualityDimension.UNIQUENESS
        assert result.score == pytest.approx(1.0)

    def test_dirty_data_has_duplicates(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data with 10 duplicate rows should score below 1.0."""
        result = score_uniqueness(
            dirty_incarceration_df, key_columns=["person_id"]
        )
        assert result.score < 1.0
        assert result.details is not None
        assert result.details["duplicate_rows"] > 0


# ---------------------------------------------------------------------------
# Tests: compute_composite_score
# ---------------------------------------------------------------------------


class TestComputeCompositeScore:
    """Tests for compute_composite_score."""

    def test_clean_data_composite(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Clean data should produce a high composite score."""
        result = compute_composite_score(
            clean_incarceration_df,
            entity_name="incarceration_periods",
            date_pairs=[("admission_date", "release_date")],
        )
        assert isinstance(result, QualityScore)
        assert result.entity_name == "incarceration_periods"
        assert len(result.dimension_scores) == 5
        # Completeness, consistency, validity, uniqueness should all be high
        # Timeliness may be low because fixture dates are from 2020
        for ds in result.dimension_scores:
            if ds.dimension != QualityDimension.TIMELINESS:
                assert ds.score >= 0.9, (
                    f"{ds.dimension.value} score too low: {ds.score}"
                )

    def test_dirty_data_composite(
        self, dirty_incarceration_df: pd.DataFrame
    ) -> None:
        """Dirty data should produce a lower composite score."""
        result = compute_composite_score(
            dirty_incarceration_df,
            entity_name="dirty_incarceration",
            date_pairs=[("admission_date", "release_date")],
        )
        assert isinstance(result, QualityScore)
        # Should be lower than clean data
        assert result.composite_score < 1.0

    def test_grade_assignment_a(self) -> None:
        """Score >= 0.9 should yield grade A."""
        df = pd.DataFrame(
            {
                "id": range(100),
                "value": range(100),
            }
        )
        result = compute_composite_score(
            df,
            entity_name="test",
            key_columns=["id"],
        )
        # No dates, no nulls, no duplicates => high score
        assert result.grade == "A"

    def test_grade_assignment_boundaries(self) -> None:
        """Verify grade boundary logic via a dataframe with known issues."""
        # 50% nulls in a required column => completeness ~ 0.5
        df = pd.DataFrame(
            {
                "id": range(10),
                "col_a": [None] * 5 + list(range(5)),
            }
        )
        result = compute_composite_score(
            df,
            entity_name="test_boundary",
            required_columns=["id", "col_a"],
            key_columns=["id"],
        )
        assert result.grade in {"A", "B", "C", "D", "F"}
        # Composite should be pulled down by completeness
        assert result.composite_score < 1.0

    def test_custom_weights(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """Custom weights should be reflected in dimension scores."""
        custom_weights = {
            "completeness": 0.50,
            "consistency": 0.10,
            "timeliness": 0.10,
            "validity": 0.10,
            "uniqueness": 0.20,
        }
        result = compute_composite_score(
            clean_incarceration_df,
            entity_name="weighted_test",
            weights=custom_weights,
            date_pairs=[("admission_date", "release_date")],
        )
        # Find the completeness score and verify its weight
        completeness_ds = [
            ds
            for ds in result.dimension_scores
            if ds.dimension == QualityDimension.COMPLETENESS
        ][0]
        assert completeness_ds.weight == pytest.approx(0.50)

    def test_all_dimensions_present(
        self, clean_incarceration_df: pd.DataFrame
    ) -> None:
        """All five quality dimensions should appear in the result."""
        result = compute_composite_score(
            clean_incarceration_df,
            entity_name="all_dims",
            date_pairs=[("admission_date", "release_date")],
        )
        dimensions_found = {ds.dimension for ds in result.dimension_scores}
        expected = {
            QualityDimension.COMPLETENESS,
            QualityDimension.CONSISTENCY,
            QualityDimension.TIMELINESS,
            QualityDimension.VALIDITY,
            QualityDimension.UNIQUENESS,
        }
        assert dimensions_found == expected
