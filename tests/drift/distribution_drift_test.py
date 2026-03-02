"""Tests for distribution drift detection functions.

Covers:
- Kolmogorov-Smirnov test on identical distributions (p ~ 1).
- Kolmogorov-Smirnov test on shifted distributions (p ~ 0).
- Categorical chi-squared drift detection.
- Drift severity classification across all thresholds.
- Edge cases: NaN handling, single-category data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cj_data_quality.drift.distribution_drift import (
    classify_drift_severity,
    detect_categorical_drift,
    detect_numeric_drift,
)
from cj_data_quality.types import DriftResult, DriftSeverity


# ---------------------------------------------------------------------------
# classify_drift_severity
# ---------------------------------------------------------------------------


class TestClassifyDriftSeverity:
    """Tests for the severity classification function."""

    def test_none_severity_above_warning(self) -> None:
        assert classify_drift_severity(0.10) == DriftSeverity.NONE
        assert classify_drift_severity(0.50) == DriftSeverity.NONE
        assert classify_drift_severity(1.0) == DriftSeverity.NONE

    def test_none_severity_at_boundary(self) -> None:
        # p > 0.05 is NONE; p = 0.06 should be NONE
        assert classify_drift_severity(0.06) == DriftSeverity.NONE

    def test_low_severity(self) -> None:
        assert classify_drift_severity(0.05) == DriftSeverity.LOW
        assert classify_drift_severity(0.02) == DriftSeverity.LOW
        assert classify_drift_severity(0.011) == DriftSeverity.LOW

    def test_medium_severity(self) -> None:
        assert classify_drift_severity(0.01) == DriftSeverity.MEDIUM
        assert classify_drift_severity(0.005) == DriftSeverity.MEDIUM
        assert classify_drift_severity(0.0011) == DriftSeverity.MEDIUM

    def test_high_severity(self) -> None:
        assert classify_drift_severity(0.001) == DriftSeverity.HIGH
        assert classify_drift_severity(0.0005) == DriftSeverity.HIGH
        assert classify_drift_severity(0.00011) == DriftSeverity.HIGH

    def test_critical_severity(self) -> None:
        assert classify_drift_severity(0.0001) == DriftSeverity.CRITICAL
        assert classify_drift_severity(0.00001) == DriftSeverity.CRITICAL
        assert classify_drift_severity(0.0) == DriftSeverity.CRITICAL


# ---------------------------------------------------------------------------
# detect_numeric_drift
# ---------------------------------------------------------------------------


class TestDetectNumericDrift:
    """Tests for KS-based numeric drift detection."""

    def test_identical_distributions_high_p_value(self) -> None:
        """KS test on samples from the same distribution should yield p ~ 1."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(100, 15, 500))
        comp = pd.Series(np.random.normal(100, 15, 500))

        result: DriftResult = detect_numeric_drift(
            reference=ref,
            comparison=comp,
            column_name="test_col",
            ref_period="2022-Q1",
            comp_period="2022-Q2",
        )

        assert isinstance(result, DriftResult)
        assert result.test_name == "ks_2samp"
        assert result.p_value > 0.05
        assert result.severity == DriftSeverity.NONE
        assert result.column_name == "test_col"
        assert result.reference_period == "2022-Q1"
        assert result.comparison_period == "2022-Q2"

    def test_shifted_distributions_low_p_value(self) -> None:
        """KS test on clearly different distributions should yield p ~ 0."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(100, 10, 500))
        comp = pd.Series(np.random.normal(200, 10, 500))

        result: DriftResult = detect_numeric_drift(
            reference=ref,
            comparison=comp,
            column_name="shifted_col",
            ref_period="2022-Q1",
            comp_period="2022-Q2",
        )

        assert result.p_value < 0.0001
        assert result.severity == DriftSeverity.CRITICAL
        assert result.statistic > 0.5  # Large KS statistic

    def test_slightly_shifted_distributions(self) -> None:
        """A small shift should produce intermediate severity."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(100, 15, 300))
        # Small shift: 0.5 standard deviations
        comp = pd.Series(np.random.normal(107.5, 15, 300))

        result: DriftResult = detect_numeric_drift(
            reference=ref,
            comparison=comp,
            column_name="slight_shift",
            ref_period="2022-Q1",
            comp_period="2022-Q2",
        )

        # Should detect some drift but not necessarily CRITICAL
        assert result.p_value < 0.05
        assert result.severity != DriftSeverity.NONE

    def test_nan_values_are_dropped(self) -> None:
        """NaN values in either series should be dropped before the test."""
        np.random.seed(42)
        ref_data = list(np.random.normal(100, 15, 100))
        comp_data = list(np.random.normal(100, 15, 100))

        # Inject NaNs
        ref_data[0] = np.nan
        ref_data[10] = np.nan
        comp_data[5] = np.nan

        ref = pd.Series(ref_data)
        comp = pd.Series(comp_data)

        result: DriftResult = detect_numeric_drift(
            reference=ref,
            comparison=comp,
            column_name="nan_col",
            ref_period="2022-Q1",
            comp_period="2022-Q2",
        )

        assert isinstance(result, DriftResult)
        assert result.details is not None
        assert result.details["ref_n"] == 98.0  # 100 - 2 NaNs
        assert result.details["comp_n"] == 99.0  # 100 - 1 NaN

    def test_empty_reference_raises(self) -> None:
        """All-NaN reference series should raise ValueError."""
        ref = pd.Series([np.nan, np.nan, np.nan])
        comp = pd.Series([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Reference series"):
            detect_numeric_drift(ref, comp, "col", "p1", "p2")

    def test_empty_comparison_raises(self) -> None:
        """All-NaN comparison series should raise ValueError."""
        ref = pd.Series([1.0, 2.0, 3.0])
        comp = pd.Series([np.nan, np.nan])

        with pytest.raises(ValueError, match="Comparison series"):
            detect_numeric_drift(ref, comp, "col", "p1", "p2")

    def test_details_contain_summary_stats(self) -> None:
        """Result details should include mean, std, and count for both series."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 5, 100))
        comp = pd.Series(np.random.normal(50, 5, 100))

        result: DriftResult = detect_numeric_drift(
            ref, comp, "stats_col", "A", "B"
        )

        assert result.details is not None
        expected_keys = {"ref_mean", "ref_std", "comp_mean", "comp_std", "ref_n", "comp_n"}
        assert expected_keys == set(result.details.keys())


# ---------------------------------------------------------------------------
# detect_categorical_drift
# ---------------------------------------------------------------------------


class TestDetectCategoricalDrift:
    """Tests for chi-squared categorical drift detection."""

    def test_identical_categorical_distributions(self) -> None:
        """Same distribution should yield high p-value."""
        np.random.seed(42)
        categories = ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]
        probs = [0.4, 0.25, 0.2, 0.1, 0.05]

        ref = pd.Series(np.random.choice(categories, size=500, p=probs))
        comp = pd.Series(np.random.choice(categories, size=500, p=probs))

        result: DriftResult = detect_categorical_drift(
            reference=ref,
            comparison=comp,
            column_name="race",
            ref_period="2022-Q1",
            comp_period="2022-Q2",
        )

        assert isinstance(result, DriftResult)
        assert result.test_name == "chi2_contingency"
        assert result.p_value > 0.05
        assert result.severity == DriftSeverity.NONE

    def test_shifted_categorical_distributions(self) -> None:
        """Clearly different categorical distributions should yield low p-value."""
        np.random.seed(42)
        categories = ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]

        ref = pd.Series(
            np.random.choice(categories, size=500, p=[0.4, 0.25, 0.2, 0.1, 0.05])
        )
        # Dramatically different distribution
        comp = pd.Series(
            np.random.choice(categories, size=500, p=[0.05, 0.1, 0.2, 0.25, 0.4])
        )

        result: DriftResult = detect_categorical_drift(
            reference=ref,
            comparison=comp,
            column_name="race",
            ref_period="2022-Q1",
            comp_period="2022-Q2",
        )

        assert result.p_value < 0.001
        assert result.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_low_frequency_categories_bucketed_to_other(self) -> None:
        """Categories with < 5 total observations should be bucketed to OTHER."""
        # Reference: mostly A and B, with rare C (2 occurrences total)
        ref = pd.Series(["A"] * 50 + ["B"] * 48 + ["C"] * 1)
        comp = pd.Series(["A"] * 50 + ["B"] * 49 + ["C"] * 1)

        result: DriftResult = detect_categorical_drift(
            ref, comp, "cat_col", "p1", "p2"
        )

        assert isinstance(result, DriftResult)
        # C has total count = 2 < 5, so it gets bucketed to OTHER
        # But the test should still run successfully
        assert result.test_name == "chi2_contingency"

    def test_nan_values_dropped(self) -> None:
        """NaN values should be excluded from categorical drift analysis."""
        ref = pd.Series(["A"] * 30 + ["B"] * 30 + [None] * 10)
        comp = pd.Series(["A"] * 30 + ["B"] * 30 + [None] * 5)

        result: DriftResult = detect_categorical_drift(
            ref, comp, "cat_nan", "p1", "p2"
        )

        assert result.details is not None
        assert result.details["ref_n"] == 60.0
        assert result.details["comp_n"] == 60.0

    def test_empty_reference_raises(self) -> None:
        """All-NaN reference should raise ValueError."""
        ref = pd.Series([None, None, None])
        comp = pd.Series(["A", "B", "C"])

        with pytest.raises(ValueError, match="Reference series"):
            detect_categorical_drift(ref, comp, "col", "p1", "p2")

    def test_single_category_returns_none_severity(self) -> None:
        """A single category in both series should return NONE severity."""
        ref = pd.Series(["A"] * 100)
        comp = pd.Series(["A"] * 100)

        result: DriftResult = detect_categorical_drift(
            ref, comp, "single_cat", "p1", "p2"
        )

        assert result.severity == DriftSeverity.NONE
        assert result.p_value == 1.0
