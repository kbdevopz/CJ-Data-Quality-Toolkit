"""Frozen attrs data classes for the CJ Data Quality toolkit.

All types are immutable (frozen=True) following Recidiviz conventions.
"""

from __future__ import annotations

import enum
from datetime import date

import attr


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ColumnDataType(enum.Enum):
    """Semantic type of a column, inferred from its data."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    FREE_TEXT = "free_text"
    UNKNOWN = "unknown"


class DriftSeverity(enum.Enum):
    """Severity level for distribution drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(enum.Enum):
    """Type of anomaly detected."""

    ZSCORE = "zscore"
    IQR = "iqr"
    ROLLING_WINDOW = "rolling_window"
    MISSING_PERIOD = "missing_period"
    SUDDEN_SPIKE = "sudden_spike"
    SUDDEN_DROP = "sudden_drop"


class QualityDimension(enum.Enum):
    """Data quality dimension for scoring."""

    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


# ---------------------------------------------------------------------------
# Profile types
# ---------------------------------------------------------------------------


@attr.s(frozen=True, kw_only=True)
class NumericStats:
    """Summary statistics for a numeric column."""

    mean: float = attr.ib()
    median: float = attr.ib()
    std: float = attr.ib()
    min_value: float = attr.ib()
    max_value: float = attr.ib()
    p25: float = attr.ib()
    p75: float = attr.ib()
    skewness: float | None = attr.ib(default=None)


@attr.s(frozen=True, kw_only=True)
class TemporalStats:
    """Summary statistics for a temporal column."""

    min_date: date = attr.ib()
    max_date: date = attr.ib()
    date_range_days: int = attr.ib()
    most_common_day_of_week: int | None = attr.ib(default=None)
    most_common_month: int | None = attr.ib(default=None)


@attr.s(frozen=True, kw_only=True)
class ColumnProfile:
    """Complete profile of a single column."""

    column_name: str = attr.ib()
    inferred_type: ColumnDataType = attr.ib()
    total_count: int = attr.ib()
    null_count: int = attr.ib()
    null_rate: float = attr.ib()
    distinct_count: int = attr.ib()
    cardinality_ratio: float = attr.ib()
    top_values: list[tuple[str, int]] = attr.ib(factory=list)
    numeric_stats: NumericStats | None = attr.ib(default=None)
    temporal_stats: TemporalStats | None = attr.ib(default=None)


@attr.s(frozen=True, kw_only=True)
class TableProfile:
    """Aggregated profile of an entire table."""

    table_name: str = attr.ib()
    row_count: int = attr.ib()
    column_count: int = attr.ib()
    column_profiles: list[ColumnProfile] = attr.ib(factory=list)
    duplicate_row_count: int = attr.ib(default=0)
    duplicate_rate: float = attr.ib(default=0.0)
    overall_null_rate: float = attr.ib(default=0.0)


# ---------------------------------------------------------------------------
# Drift types
# ---------------------------------------------------------------------------


@attr.s(frozen=True, kw_only=True)
class DriftResult:
    """Result of a distribution drift test between two periods."""

    column_name: str = attr.ib()
    test_name: str = attr.ib()
    statistic: float = attr.ib()
    p_value: float = attr.ib()
    severity: DriftSeverity = attr.ib()
    reference_period: str = attr.ib()
    comparison_period: str = attr.ib()
    details: dict[str, float] | None = attr.ib(default=None)


# ---------------------------------------------------------------------------
# Anomaly types
# ---------------------------------------------------------------------------


@attr.s(frozen=True, kw_only=True)
class AnomalyResult:
    """A single detected anomaly."""

    metric_name: str = attr.ib()
    anomaly_type: AnomalyType = attr.ib()
    timestamp: date = attr.ib()
    observed_value: float = attr.ib()
    expected_value: float = attr.ib()
    deviation: float = attr.ib()
    threshold: float = attr.ib()
    state_code: str | None = attr.ib(default=None)


# ---------------------------------------------------------------------------
# Coverage types
# ---------------------------------------------------------------------------


@attr.s(frozen=True, kw_only=True)
class CoverageCell:
    """A single cell in a coverage matrix (state x metric)."""

    state_code: str = attr.ib()
    metric_name: str = attr.ib()
    completeness: float = attr.ib()  # 0.0 to 1.0
    total_expected: int = attr.ib()
    total_present: int = attr.ib()
    is_gap: bool = attr.ib(default=False)


@attr.s(frozen=True, kw_only=True)
class EquityCoverage:
    """Demographic coverage analysis for a single state."""

    state_code: str = attr.ib()
    field_name: str = attr.ib()
    completeness: float = attr.ib()
    distinct_values: int = attr.ib()
    most_common_value: str | None = attr.ib(default=None)
    most_common_rate: float | None = attr.ib(default=None)
    disparity_index: float | None = attr.ib(default=None)


# ---------------------------------------------------------------------------
# Quality score types
# ---------------------------------------------------------------------------


@attr.s(frozen=True, kw_only=True)
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: QualityDimension = attr.ib()
    score: float = attr.ib()  # 0.0 to 1.0
    weight: float = attr.ib()
    details: dict[str, float] | None = attr.ib(default=None)


@attr.s(frozen=True, kw_only=True)
class QualityScore:
    """Composite quality score across multiple dimensions."""

    entity_name: str = attr.ib()
    composite_score: float = attr.ib()
    dimension_scores: list[DimensionScore] = attr.ib(factory=list)
    grade: str = attr.ib(default="")  # A/B/C/D/F
