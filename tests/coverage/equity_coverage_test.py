"""Tests for the equity coverage module.

Uses the ``demographic_coverage_df`` fixture defined in ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.coverage.equity_coverage import (
    analyze_demographic_completeness,
    compute_equity_disparity_index,
)
from cj_data_quality.types import EquityCoverage


# ---- analyze_demographic_completeness --------------------------------------


class TestAnalyzeDemographicCompleteness:
    """Tests for :func:`analyze_demographic_completeness`."""

    def test_returns_equity_coverage_instances(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """All returned items must be EquityCoverage instances."""
        results = analyze_demographic_completeness(demographic_coverage_df)
        assert len(results) > 0
        for item in results:
            assert isinstance(item, EquityCoverage)

    def test_result_count(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Should return one entry per state-field pair."""
        fields = ["race", "ethnicity", "sex", "age"]
        results = analyze_demographic_completeness(
            demographic_coverage_df, demographic_fields=fields
        )
        # 5 states x 4 fields = 20
        assert len(results) == 20

    def test_ca_full_completeness(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_CA has 100% coverage for all demographic fields."""
        results = analyze_demographic_completeness(
            demographic_coverage_df,
            demographic_fields=["race", "ethnicity", "sex", "age"],
        )
        ca_results = [r for r in results if r.state_code == "US_CA"]
        for r in ca_results:
            assert r.completeness == pytest.approx(
                1.0
            ), f"US_CA/{r.field_name} should be 1.0"

    def test_tx_race_partial(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_TX race completeness should be around 60% (40% missing)."""
        results = analyze_demographic_completeness(
            demographic_coverage_df, demographic_fields=["race"]
        )
        tx_race = [
            r
            for r in results
            if r.state_code == "US_TX" and r.field_name == "race"
        ]
        assert len(tx_race) == 1
        assert tx_race[0].completeness < 1.0
        assert tx_race[0].completeness > 0.3

    def test_tx_ethnicity_partial(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_TX ethnicity completeness should be around 40% (60% missing)."""
        results = analyze_demographic_completeness(
            demographic_coverage_df, demographic_fields=["ethnicity"]
        )
        tx_eth = [
            r
            for r in results
            if r.state_code == "US_TX" and r.field_name == "ethnicity"
        ]
        assert len(tx_eth) == 1
        assert tx_eth[0].completeness < 0.7
        assert tx_eth[0].completeness > 0.1

    def test_fl_ethnicity_low(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_FL ethnicity completeness should be around 20% (80% missing)."""
        results = analyze_demographic_completeness(
            demographic_coverage_df, demographic_fields=["ethnicity"]
        )
        fl_eth = [
            r
            for r in results
            if r.state_code == "US_FL" and r.field_name == "ethnicity"
        ]
        assert len(fl_eth) == 1
        assert fl_eth[0].completeness < 0.5

    def test_distinct_values_ny_race(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """US_NY race should have exactly 2 distinct values (WHITE, BLACK)."""
        results = analyze_demographic_completeness(
            demographic_coverage_df, demographic_fields=["race"]
        )
        ny_race = [
            r
            for r in results
            if r.state_code == "US_NY" and r.field_name == "race"
        ]
        assert len(ny_race) == 1
        assert ny_race[0].distinct_values == 2

    def test_most_common_value_present(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """For complete fields the most common value and rate must be set."""
        results = analyze_demographic_completeness(
            demographic_coverage_df, demographic_fields=["sex"]
        )
        for r in results:
            assert r.most_common_value is not None
            assert r.most_common_rate is not None
            assert 0.0 < r.most_common_rate <= 1.0

    def test_defaults_to_cj_demographic_fields(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """When demographic_fields is None, uses CJ_DEMOGRAPHIC_FIELDS present in df."""
        results = analyze_demographic_completeness(demographic_coverage_df)
        field_names = {r.field_name for r in results}
        # The fixture has race, ethnicity, sex, age (age_group absent)
        assert field_names == {"race", "ethnicity", "sex", "age"}

    def test_completeness_bounded(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """All completeness values must be in [0.0, 1.0]."""
        results = analyze_demographic_completeness(demographic_coverage_df)
        for r in results:
            assert 0.0 <= r.completeness <= 1.0


# ---- compute_equity_disparity_index ----------------------------------------


class TestComputeEquityDisparityIndex:
    """Tests for :func:`compute_equity_disparity_index`."""

    def test_returns_per_state(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """Should return one entry per state present in the data."""
        result = compute_equity_disparity_index(
            demographic_coverage_df, "state_code", "race", "age"
        )
        assert isinstance(result, dict)
        # 5 states (some may be absent if all race/age null, but fixture
        # guarantees at least TX has some non-null race)
        assert len(result) >= 4

    def test_disparity_non_negative(
        self, demographic_coverage_df: pd.DataFrame
    ) -> None:
        """All disparity indices must be >= 0."""
        result = compute_equity_disparity_index(
            demographic_coverage_df, "state_code", "race", "age"
        )
        for val in result.values():
            assert val >= 0.0

    def test_uniform_metric_zero_disparity(self) -> None:
        """When all groups have the same metric mean, disparity should be 0."""
        df = pd.DataFrame(
            {
                "state_code": ["US_XX"] * 6,
                "race": ["A", "A", "B", "B", "C", "C"],
                "metric": [10, 10, 10, 10, 10, 10],
            }
        )
        result = compute_equity_disparity_index(
            df, "state_code", "race", "metric"
        )
        assert result["US_XX"] == pytest.approx(0.0)

    def test_high_disparity_detected(self) -> None:
        """When groups have very different means, disparity should be high."""
        df = pd.DataFrame(
            {
                "state_code": ["US_XX"] * 6,
                "race": ["A", "A", "B", "B", "C", "C"],
                "metric": [1, 1, 100, 100, 200, 200],
            }
        )
        result = compute_equity_disparity_index(
            df, "state_code", "race", "metric"
        )
        assert result["US_XX"] > 0.5

    def test_single_group_zero_disparity(self) -> None:
        """A state with only one group should have disparity 0.0."""
        df = pd.DataFrame(
            {
                "state_code": ["US_XX"] * 4,
                "race": ["A", "A", "A", "A"],
                "metric": [10, 20, 30, 40],
            }
        )
        result = compute_equity_disparity_index(
            df, "state_code", "race", "metric"
        )
        assert result["US_XX"] == pytest.approx(0.0)

    def test_null_rows_excluded(self) -> None:
        """Rows with null group or metric values should be excluded."""
        df = pd.DataFrame(
            {
                "state_code": ["US_XX"] * 6,
                "race": ["A", "A", "B", "B", None, "C"],
                "metric": [10, 10, 10, 10, 10, None],
            }
        )
        result = compute_equity_disparity_index(
            df, "state_code", "race", "metric"
        )
        # Only groups A and B contribute (C has null metric)
        assert result["US_XX"] == pytest.approx(0.0)

    def test_multiple_states_independent(self) -> None:
        """Disparity is computed independently per state."""
        df = pd.DataFrame(
            {
                "state_code": ["US_AA"] * 4 + ["US_BB"] * 4,
                "race": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
                "metric": [10, 10, 10, 10, 10, 10, 100, 100],
            }
        )
        result = compute_equity_disparity_index(
            df, "state_code", "race", "metric"
        )
        assert result["US_AA"] == pytest.approx(0.0)
        assert result["US_BB"] > 0.0
