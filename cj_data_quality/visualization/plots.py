"""Recidiviz-styled visualization functions for CJ data quality.

All charts use the Recidiviz 11-color palette and consistent styling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from cj_data_quality.constants import (
    NULL_RATE_CRITICAL,
    NULL_RATE_WARNING,
    RECIDIVIZ_BG_LIGHT,
    RECIDIVIZ_COLORS,
    RECIDIVIZ_DARK_TEAL,
    RECIDIVIZ_TEXT_DARK,
)
from cj_data_quality.types import ColumnProfile, QualityScore, TableProfile
from cj_data_quality.visualization._style import apply_style as _apply_style


def _color_by_null_rate(rate: float) -> str:
    """Return a color based on null rate severity."""
    if rate >= NULL_RATE_CRITICAL:
        return "#FF6B4D"  # Coral (critical)
    if rate >= NULL_RATE_WARNING:
        return "#FFB84D"  # Amber (warning)
    return "#25B894"  # Green (good)


def plot_null_rate_bars(
    profiles: list[ColumnProfile],
    title: str = "Column Null Rates",
    figsize: tuple[int, int] = (12, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart of null rates per column, color-coded by severity."""
    _apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]
        assert fig is not None

    if not profiles:
        ax.text(0.5, 0.5, "No column profiles available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
        fig.tight_layout()
        return fig

    names = [p.column_name for p in profiles]
    rates = [p.null_rate for p in profiles]
    colors = [_color_by_null_rate(r) for r in rates]

    bars = ax.barh(names, rates, color=colors, edgecolor="none")
    ax.set_xlabel("Null Rate")
    ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
    ax.set_xlim(0, max(1.0, max(rates) * 1.1))

    # Add threshold lines
    ax.axvline(x=NULL_RATE_WARNING, color="#FFB84D", linestyle="--", alpha=0.7, label="Warning")
    ax.axvline(x=NULL_RATE_CRITICAL, color="#FF6B4D", linestyle="--", alpha=0.7, label="Critical")
    ax.legend(loc="lower right", fontsize=9)

    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_profile_summary(
    table_profile: TableProfile,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Summary dashboard of a table profile: null rates + cardinality."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    profiles = table_profile.column_profiles

    if not profiles:
        axes[0].text(0.5, 0.5, "No columns to profile", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        axes[1].text(0.5, 0.5, "No columns to profile", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        fig.suptitle(f"Profile: {table_profile.table_name}", fontsize=15,
                     fontweight="bold", color=RECIDIVIZ_DARK_TEAL)
        fig.tight_layout()
        return fig

    # Left: null rates
    names = [p.column_name for p in profiles]
    null_rates = [p.null_rate for p in profiles]
    colors = [_color_by_null_rate(r) for r in null_rates]
    axes[0].barh(names, null_rates, color=colors)
    axes[0].set_title("Null Rates", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1.0)
    axes[0].invert_yaxis()

    # Right: cardinality
    cardinalities = [p.distinct_count for p in profiles]
    axes[1].barh(names, cardinalities, color=RECIDIVIZ_COLORS[1])
    axes[1].set_title("Distinct Value Counts", fontsize=13, fontweight="bold")
    axes[1].invert_yaxis()

    fig.suptitle(
        f"Profile: {table_profile.table_name} ({table_profile.row_count:,} rows)",
        fontsize=15,
        fontweight="bold",
        color=RECIDIVIZ_DARK_TEAL,
    )
    fig.tight_layout()
    return fig


def plot_drift_timeline(
    drift_df: pd.DataFrame,
    title: str = "Distribution Drift Over Time",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Line chart of drift p-values over consecutive period pairs.

    Args:
        drift_df: DataFrame with columns 'period_pair' and 'p_value'.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    if drift_df.empty:
        ax.text(0.5, 0.5, "No drift data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
        fig.tight_layout()
        return fig

    ax.plot(
        drift_df["period_pair"],
        drift_df["p_value"],
        marker="o",
        color=RECIDIVIZ_COLORS[0],
        linewidth=2,
        markersize=6,
    )

    # Threshold bands
    ax.axhline(y=0.05, color="#FFB84D", linestyle="--", alpha=0.7, label="p=0.05")
    ax.axhline(y=0.001, color="#FF6B4D", linestyle="--", alpha=0.7, label="p=0.001")

    ax.set_ylabel("p-value")
    ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")  # type: ignore[attr-defined]
    fig.tight_layout()
    return fig


def plot_anomaly_scatter(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    anomaly_indices: list[int],
    title: str = "Anomaly Detection",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Scatter plot highlighting anomaly points in a time series.

    Args:
        df: Full time series DataFrame.
        date_col: Column with dates.
        value_col: Column with values.
        anomaly_indices: Row indices of detected anomalies.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        df[date_col], df[value_col],
        color=RECIDIVIZ_COLORS[0], linewidth=1.5, alpha=0.8,
    )

    if anomaly_indices:
        anomaly_df = df.iloc[anomaly_indices]
        ax.scatter(
            anomaly_df[date_col],
            anomaly_df[value_col],
            color="#FF6B4D",
            s=80,
            zorder=5,
            label=f"Anomalies ({len(anomaly_indices)})",
            edgecolors="white",
            linewidths=1,
        )

    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_quality_scorecard(
    score: QualityScore,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Horizontal bar chart of dimension scores with composite grade."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    if not score.dimension_scores:
        ax.text(0.5, 0.5, "No dimension scores available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        ax.set_title(score.entity_name, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
        fig.tight_layout()
        return fig

    dims = [ds.dimension.value.title() for ds in score.dimension_scores]
    scores = [ds.score for ds in score.dimension_scores]

    colors = []
    for s in scores:
        if s >= 0.85:
            colors.append("#25B894")
        elif s >= 0.60:
            colors.append("#FFB84D")
        else:
            colors.append("#FF6B4D")

    ax.barh(dims, scores, color=colors, edgecolor="none")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")

    # Add score labels
    for i, (s, name) in enumerate(zip(scores, dims)):
        ax.text(s + 0.02, i, f"{s:.2f}", va="center", fontsize=10)

    grade_color = {
        "A": "#25B894", "B": "#00A5CF", "C": "#FFB84D", "D": "#FF6B4D", "F": "#C44D97"
    }
    ax.set_title(
        f"{score.entity_name} — Composite: {score.composite_score:.2f} (Grade: {score.grade})",
        fontsize=14,
        fontweight="bold",
        color=grade_color.get(score.grade, RECIDIVIZ_TEXT_DARK),
    )
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
