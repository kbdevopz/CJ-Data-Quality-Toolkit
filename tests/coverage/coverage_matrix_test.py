"""Tests for the coverage matrix module.

Uses the ``demographic_coverage_df`` fixture defined in ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.coverage.coverage_matrix import (
    build_coverage_matrix,
    identify_coverage_gaps,
    summarize_coverage,
)
from cj_data_quality.types import CoverageCell


# ---- build_coverage_matrix -------------------------------------------------


class TestBuildCoverageMatrix:
    """Tests for :func:`build_coverage_matrix`."""

    def test_matrix_shape(self, demographic_coverage_df: pd.DataFrame) -> None:
        """Matrix should have one row per state and one column per metric."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        # 5 states: US_CA, US_TX, US_NY, US_FL, US_OH
        assert matrix.shape == (5, 4)
        assert set(matrix.columns) == set(metric_cols)

    def test_full_coverage_states(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_CA and US_OH have full coverage for all fields."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        for state in ["US_CA", "US_OH"]:
            for metric in metric_cols:
                assert matrix.loc[state, metric] == pytest.approx(
                    1.0
                ), f"{state}/{metric} should be 1.0"

    def test_partial_coverage_tx(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_TX has ~60% race completeness and ~40% ethnicity completeness."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        # TX race ~60% (40% missing), ethnicity ~40% (60% missing)
        assert matrix.loc["US_TX", "race"] < 1.0
        assert matrix.loc["US_TX", "race"] > 0.3
        assert matrix.loc["US_TX", "ethnicity"] < 0.7
        assert matrix.loc["US_TX", "ethnicity"] > 0.1
        # sex and age should be complete
        assert matrix.loc["US_TX", "sex"] == pytest.approx(1.0)
        assert matrix.loc["US_TX", "age"] == pytest.approx(1.0)

    def test_index_name(self, demographic_coverage_df: pd.DataFrame) -> None:
        """Matrix index should be named after the state column."""
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", ["race"]
        )
        assert matrix.index.name == "state_code"

    def test_values_bounded(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """All completeness values must be between 0.0 and 1.0."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        assert (matrix.values >= 0.0).all()
        assert (matrix.values <= 1.0).all()


# ---- identify_coverage_gaps ------------------------------------------------


class TestIdentifyCoverageGaps:
    """Tests for :func:`identify_coverage_gaps`."""

    def test_known_gaps_detected(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """TX race/ethnicity and FL ethnicity should be flagged at threshold 0.8."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        gaps = identify_coverage_gaps(matrix, threshold=0.8)

        gap_keys = {(g.state_code, g.metric_name) for g in gaps}
        # TX race (~60%) and TX ethnicity (~40%) should be gaps
        assert ("US_TX", "race") in gap_keys
        assert ("US_TX", "ethnicity") in gap_keys
        # FL ethnicity (~20%) should be a gap
        assert ("US_FL", "ethnicity") in gap_keys

    def test_no_gaps_for_full_states(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """States with complete data should not appear in gaps."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        gaps = identify_coverage_gaps(matrix, threshold=0.8)

        gap_states = {g.state_code for g in gaps}
        assert "US_CA" not in gap_states
        assert "US_OH" not in gap_states

    def test_all_gaps_have_is_gap_true(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Every returned CoverageCell must have is_gap=True."""
        metric_cols = ["race", "ethnicity"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        gaps = identify_coverage_gaps(matrix, threshold=0.8)
        for gap in gaps:
            assert gap.is_gap is True

    def test_gaps_sorted_ascending(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Gaps should be sorted by completeness ascending."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        gaps = identify_coverage_gaps(matrix, threshold=0.8)
        completeness_values = [g.completeness for g in gaps]
        assert completeness_values == sorted(completeness_values)

    def test_threshold_zero_returns_nothing(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """With threshold=0.0, no gaps should be reported."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        gaps = identify_coverage_gaps(matrix, threshold=0.0)
        assert len(gaps) == 0

    def test_threshold_one_flags_imperfect(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """With threshold=1.0, any non-perfect cell is flagged."""
        metric_cols = ["race", "ethnicity"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        gaps = identify_coverage_gaps(matrix, threshold=1.0)
        # TX and FL have imperfect coverage, so they must appear
        gap_states = {g.state_code for g in gaps}
        assert "US_TX" in gap_states
        assert "US_FL" in gap_states

    def test_return_type(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Returned objects must be CoverageCell instances."""
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", ["race"]
        )
        gaps = identify_coverage_gaps(matrix, threshold=0.8)
        for gap in gaps:
            assert isinstance(gap, CoverageCell)


# ---- summarize_coverage ----------------------------------------------------


class TestSummarizeCoverage:
    """Tests for :func:`summarize_coverage`."""

    def test_returns_dict_with_all_metrics(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Summary should contain one entry per metric."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        summary = summarize_coverage(matrix)
        assert set(summary.keys()) == set(metric_cols)

    def test_sex_and_age_fully_complete(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Sex and age are complete across all states, so mean should be 1.0."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        summary = summarize_coverage(matrix)
        assert summary["sex"] == pytest.approx(1.0)
        assert summary["age"] == pytest.approx(1.0)

    def test_race_average_below_one(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Race has gaps in TX, so the average should be below 1.0."""
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", ["race"]
        )
        summary = summarize_coverage(matrix)
        assert summary["race"] < 1.0

    def test_values_bounded(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """All summary values must be between 0.0 and 1.0."""
        metric_cols = ["race", "ethnicity", "sex", "age"]
        matrix = build_coverage_matrix(
            demographic_coverage_df, "state_code", metric_cols
        )
        summary = summarize_coverage(matrix)
        for val in summary.values():
            assert 0.0 <= val <= 1.0
