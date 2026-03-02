"""Distribution drift detection using statistical tests.

Provides Kolmogorov-Smirnov tests for numeric columns and chi-squared tests
for categorical columns, with severity classification based on p-value
thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp  # type: ignore[import-untyped]

from cj_data_quality.constants import DRIFT_P_VALUE_CRITICAL, DRIFT_P_VALUE_WARNING
from cj_data_quality.types import DriftResult, DriftSeverity


def classify_drift_severity(p_value: float) -> DriftSeverity:
    """Classify drift severity from a p-value.

    Thresholds:
        p > 0.05        -> NONE
        0.01 < p <= 0.05 -> LOW
        0.001 < p <= 0.01 -> MEDIUM
        0.0001 < p <= 0.001 -> HIGH
        p <= 0.0001     -> CRITICAL

    Args:
        p_value: The p-value from a statistical test.

    Returns:
        A DriftSeverity enum member.
    """
    if p_value > DRIFT_P_VALUE_WARNING:
        return DriftSeverity.NONE
    if p_value > 0.01:
        return DriftSeverity.LOW
    if p_value > DRIFT_P_VALUE_CRITICAL:
        return DriftSeverity.MEDIUM
    if p_value > 0.0001:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


def detect_numeric_drift(
    reference: pd.Series,
    comparison: pd.Series,
    column_name: str,
    ref_period: str,
    comp_period: str,
) -> DriftResult:
    """Detect drift in a numeric column using the Kolmogorov-Smirnov two-sample test.

    NaN values are dropped before running the test. Both series must have
    at least one non-NaN value.

    Args:
        reference: Numeric reference series (baseline period).
        comparison: Numeric comparison series (current period).
        column_name: Name of the column being tested.
        ref_period: Label for the reference period (e.g. "2022-Q1").
        comp_period: Label for the comparison period (e.g. "2022-Q2").

    Returns:
        A DriftResult with KS statistic, p-value, and severity.

    Raises:
        ValueError: If either series is empty after dropping NaNs.
    """
    ref_clean: np.ndarray = reference.dropna().to_numpy(dtype=float)
    comp_clean: np.ndarray = comparison.dropna().to_numpy(dtype=float)

    if len(ref_clean) == 0:
        raise ValueError(
            f"Reference series for '{column_name}' is empty after dropping NaNs."
        )
    if len(comp_clean) == 0:
        raise ValueError(
            f"Comparison series for '{column_name}' is empty after dropping NaNs."
        )

    statistic, p_value = ks_2samp(ref_clean, comp_clean)
    severity = classify_drift_severity(p_value)

    return DriftResult(
        column_name=column_name,
        test_name="ks_2samp",
        statistic=float(statistic),
        p_value=float(p_value),
        severity=severity,
        reference_period=ref_period,
        comparison_period=comp_period,
        details={
            "ref_mean": float(np.mean(ref_clean)),
            "ref_std": float(np.std(ref_clean, ddof=1)) if len(ref_clean) > 1 else 0.0,
            "comp_mean": float(np.mean(comp_clean)),
            "comp_std": float(np.std(comp_clean, ddof=1)) if len(comp_clean) > 1 else 0.0,
            "ref_n": float(len(ref_clean)),
            "comp_n": float(len(comp_clean)),
        },
    )


def detect_categorical_drift(
    reference: pd.Series,
    comparison: pd.Series,
    column_name: str,
    ref_period: str,
    comp_period: str,
) -> DriftResult:
    """Detect drift in a categorical column using the chi-squared test.

    Builds a contingency table from value counts. Categories with fewer than
    5 observations (across both periods combined) are bucketed into an
    ``"OTHER"`` category to satisfy chi-squared test assumptions.

    NaN values are dropped before building the contingency table.

    Args:
        reference: Categorical reference series (baseline period).
        comparison: Categorical comparison series (current period).
        column_name: Name of the column being tested.
        ref_period: Label for the reference period.
        comp_period: Label for the comparison period.

    Returns:
        A DriftResult with chi-squared statistic, p-value, and severity.

    Raises:
        ValueError: If either series is empty after dropping NaNs.
    """
    ref_clean: pd.Series = reference.dropna()
    comp_clean: pd.Series = comparison.dropna()

    if len(ref_clean) == 0:
        raise ValueError(
            f"Reference series for '{column_name}' is empty after dropping NaNs."
        )
    if len(comp_clean) == 0:
        raise ValueError(
            f"Comparison series for '{column_name}' is empty after dropping NaNs."
        )

    ref_counts: pd.Series = ref_clean.value_counts()
    comp_counts: pd.Series = comp_clean.value_counts()

    # Union of all categories
    all_categories: set[str] = set(ref_counts.index) | set(comp_counts.index)

    # Build aligned counts, defaulting missing categories to 0
    ref_aligned: dict[str, int] = {cat: int(ref_counts.get(cat, 0)) for cat in all_categories}
    comp_aligned: dict[str, int] = {cat: int(comp_counts.get(cat, 0)) for cat in all_categories}

    # Bucket low-frequency categories into "OTHER"
    # A category is low-frequency if its total count across both periods < 5
    other_ref: int = 0
    other_comp: int = 0
    keep_categories: list[str] = []

    for cat in all_categories:
        total: int = ref_aligned[cat] + comp_aligned[cat]
        if total < 5:
            other_ref += ref_aligned[cat]
            other_comp += comp_aligned[cat]
        else:
            keep_categories.append(cat)

    if other_ref > 0 or other_comp > 0:
        keep_categories.append("OTHER")
        ref_aligned["OTHER"] = other_ref
        comp_aligned["OTHER"] = other_comp

    # Sort for deterministic ordering
    keep_categories.sort()

    # Build contingency table: rows = [reference, comparison], cols = categories
    contingency: np.ndarray = np.array(
        [
            [ref_aligned[cat] for cat in keep_categories],
            [comp_aligned[cat] for cat in keep_categories],
        ]
    )

    # Need at least 2 categories for chi-squared
    if contingency.shape[1] < 2:
        return DriftResult(
            column_name=column_name,
            test_name="chi2_contingency",
            statistic=0.0,
            p_value=1.0,
            severity=DriftSeverity.NONE,
            reference_period=ref_period,
            comparison_period=comp_period,
            details={"note": 0.0},  # sentinel: fewer than 2 categories
        )

    statistic, p_value, dof, _ = chi2_contingency(contingency)
    severity = classify_drift_severity(p_value)

    return DriftResult(
        column_name=column_name,
        test_name="chi2_contingency",
        statistic=float(statistic),
        p_value=float(p_value),
        severity=severity,
        reference_period=ref_period,
        comparison_period=comp_period,
        details={
            "degrees_of_freedom": float(dof),
            "n_categories": float(len(keep_categories)),
            "ref_n": float(len(ref_clean)),
            "comp_n": float(len(comp_clean)),
        },
    )
