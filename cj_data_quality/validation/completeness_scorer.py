"""Composite data-quality scoring across five standard dimensions.

Each scorer produces a ``DimensionScore`` (frozen attrs class).  The
``compute_composite_score`` function rolls them up into a weighted
``QualityScore`` with a letter grade.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cj_data_quality.constants import (
    CJ_DATE_ORDERING_RULES,
    DEFAULT_QUALITY_WEIGHTS,
    EARLIEST_REASONABLE_DATE,
)
from cj_data_quality.types import DimensionScore, QualityDimension, QualityScore
from cj_data_quality.validation.temporal_consistency import (
    check_date_ordering,
    check_date_reasonableness,
)


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------


def score_completeness(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> DimensionScore:
    """Score completeness as ``1 - mean(null_rate)`` across *required_columns*.

    Args:
        df: Input DataFrame.
        required_columns: Columns to assess.  When ``None``, every column in
            *df* is considered required.

    Returns:
        A ``DimensionScore`` for the COMPLETENESS dimension.
    """
    if required_columns is None:
        required_columns = list(df.columns)

    null_rates: dict[str, float] = {}
    for col in required_columns:
        if col in df.columns:
            null_rates[col] = float(df[col].isna().mean())

    if not null_rates:
        score = 1.0
    else:
        score = 1.0 - sum(null_rates.values()) / len(null_rates)

    score = max(0.0, min(1.0, score))

    return DimensionScore(
        dimension=QualityDimension.COMPLETENESS,
        score=score,
        weight=DEFAULT_QUALITY_WEIGHTS["completeness"],
        details=null_rates,
    )


def score_consistency(
    df: pd.DataFrame,
    date_pairs: list[tuple[str, str]] | None = None,
) -> DimensionScore:
    """Score consistency as ``1 - mean(violation_rate)`` across date pairs.

    Args:
        df: Input DataFrame.
        date_pairs: Date ordering rules to check.  Defaults to
            ``CJ_DATE_ORDERING_RULES`` (filtered to present columns).

    Returns:
        A ``DimensionScore`` for the CONSISTENCY dimension.
    """
    ordering_df = check_date_ordering(df, date_pairs=date_pairs)

    if ordering_df.empty:
        score = 1.0
        details: dict[str, float] = {}
    else:
        details = {
            f"{row['earlier_field']}_vs_{row['later_field']}": float(
                row["violation_rate"]
            )
            for _, row in ordering_df.iterrows()
        }
        mean_violation_rate = ordering_df["violation_rate"].mean()
        score = 1.0 - float(mean_violation_rate)

    score = max(0.0, min(1.0, score))

    return DimensionScore(
        dimension=QualityDimension.CONSISTENCY,
        score=score,
        weight=DEFAULT_QUALITY_WEIGHTS["consistency"],
        details=details,
    )


def score_timeliness(
    df: pd.DataFrame,
    date_col: str,
    expected_max_age_days: int = 90,
) -> DimensionScore:
    """Score timeliness as the fraction of records within *expected_max_age_days*.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column to evaluate.
        expected_max_age_days: Maximum acceptable age of a record in days.

    Returns:
        A ``DimensionScore`` for the TIMELINESS dimension.
    """
    if date_col not in df.columns:
        return DimensionScore(
            dimension=QualityDimension.TIMELINESS,
            score=1.0,
            weight=DEFAULT_QUALITY_WEIGHTS["timeliness"],
            details={"note": 0.0},
        )

    non_null = df[date_col].dropna()
    total = len(non_null)

    if total == 0:
        return DimensionScore(
            dimension=QualityDimension.TIMELINESS,
            score=0.0,
            weight=DEFAULT_QUALITY_WEIGHTS["timeliness"],
            details={"total_checked": 0.0, "within_threshold": 0.0},
        )

    today = pd.Timestamp(date.today())
    cutoff = today - pd.Timedelta(days=expected_max_age_days)
    within_threshold = int((non_null >= cutoff).sum())
    score = within_threshold / total
    score = max(0.0, min(1.0, score))

    return DimensionScore(
        dimension=QualityDimension.TIMELINESS,
        score=score,
        weight=DEFAULT_QUALITY_WEIGHTS["timeliness"],
        details={
            "total_checked": float(total),
            "within_threshold": float(within_threshold),
            "expected_max_age_days": float(expected_max_age_days),
        },
    )


def score_validity(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
) -> DimensionScore:
    """Score validity as ``1 - unreasonable_rate`` from date reasonableness.

    Args:
        df: Input DataFrame.
        date_columns: Columns to check.  Defaults to all datetime columns.

    Returns:
        A ``DimensionScore`` for the VALIDITY dimension.
    """
    reasonableness_df = check_date_reasonableness(
        df,
        date_columns=date_columns,
        earliest=EARLIEST_REASONABLE_DATE,
    )

    if reasonableness_df.empty:
        score = 1.0
        details: dict[str, float] = {}
    else:
        details = {
            row["column"]: float(row["unreasonable_rate"])
            for _, row in reasonableness_df.iterrows()
        }
        mean_unreasonable = reasonableness_df["unreasonable_rate"].mean()
        score = 1.0 - float(mean_unreasonable)

    score = max(0.0, min(1.0, score))

    return DimensionScore(
        dimension=QualityDimension.VALIDITY,
        score=score,
        weight=DEFAULT_QUALITY_WEIGHTS["validity"],
        details=details,
    )


def score_uniqueness(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> DimensionScore:
    """Score uniqueness as ``1 - duplicate_rate``.

    Args:
        df: Input DataFrame.
        key_columns: Columns that form the natural key.  When ``None``, all
            columns are used to identify duplicates.

    Returns:
        A ``DimensionScore`` for the UNIQUENESS dimension.
    """
    total = len(df)
    if total == 0:
        return DimensionScore(
            dimension=QualityDimension.UNIQUENESS,
            score=1.0,
            weight=DEFAULT_QUALITY_WEIGHTS["uniqueness"],
            details={"total_rows": 0.0, "duplicate_rows": 0.0},
        )

    if key_columns is not None:
        duplicate_count = int(df.duplicated(subset=key_columns, keep="first").sum())
    else:
        duplicate_count = int(df.duplicated(keep="first").sum())

    duplicate_rate = duplicate_count / total
    score = 1.0 - duplicate_rate
    score = max(0.0, min(1.0, score))

    return DimensionScore(
        dimension=QualityDimension.UNIQUENESS,
        score=score,
        weight=DEFAULT_QUALITY_WEIGHTS["uniqueness"],
        details={
            "total_rows": float(total),
            "duplicate_rows": float(duplicate_count),
            "duplicate_rate": duplicate_rate,
        },
    )


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def assign_grade(score: float) -> str:
    """Map a 0-1 composite score to a letter grade."""
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"


def compute_composite_score(
    df: pd.DataFrame,
    entity_name: str,
    weights: dict[str, float] | None = None,
    *,
    date_col: str = "admission_date",
    expected_max_age_days: int = 90,
    date_pairs: list[tuple[str, str]] | None = None,
    required_columns: list[str] | None = None,
    key_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
) -> QualityScore:
    """Compute all dimension scores and a weighted composite score.

    Args:
        df: Input DataFrame to score.
        entity_name: Human-readable name for the dataset being scored.
        weights: Per-dimension weights.  Defaults to
            ``DEFAULT_QUALITY_WEIGHTS``.
        date_col: Column used for timeliness scoring.
        expected_max_age_days: Max acceptable record age for timeliness.
        date_pairs: Ordering rules for consistency scoring.
        required_columns: Columns for completeness scoring.
        key_columns: Key columns for uniqueness scoring.
        date_columns: Date columns for validity scoring.

    Returns:
        A ``QualityScore`` with per-dimension breakdowns and a letter grade.
    """
    if weights is None:
        weights = dict(DEFAULT_QUALITY_WEIGHTS)

    # Compute each dimension --------------------------------------------------
    completeness = score_completeness(df, required_columns=required_columns)
    consistency = score_consistency(df, date_pairs=date_pairs)
    timeliness = score_timeliness(
        df, date_col=date_col, expected_max_age_days=expected_max_age_days
    )
    validity = score_validity(df, date_columns=date_columns)
    uniqueness = score_uniqueness(df, key_columns=key_columns)

    # Override weights if caller provided custom ones -------------------------
    dimension_scores: list[DimensionScore] = []
    for dim_score, dim_key in [
        (completeness, "completeness"),
        (consistency, "consistency"),
        (timeliness, "timeliness"),
        (validity, "validity"),
        (uniqueness, "uniqueness"),
    ]:
        w = weights.get(dim_key, dim_score.weight)
        # Rebuild with the (possibly updated) weight
        dimension_scores.append(
            DimensionScore(
                dimension=dim_score.dimension,
                score=dim_score.score,
                weight=w,
                details=dim_score.details,
            )
        )

    # Weighted average --------------------------------------------------------
    total_weight = sum(ds.weight for ds in dimension_scores)
    if total_weight > 0:
        composite = sum(ds.score * ds.weight for ds in dimension_scores) / total_weight
    else:
        composite = 0.0

    composite = max(0.0, min(1.0, composite))
    grade = assign_grade(composite)

    return QualityScore(
        entity_name=entity_name,
        composite_score=composite,
        dimension_scores=dimension_scores,
        grade=grade,
    )
