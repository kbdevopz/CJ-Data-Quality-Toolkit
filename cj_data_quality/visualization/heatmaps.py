"""Heatmap visualizations for coverage and quality matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cj_data_quality.constants import (
    RECIDIVIZ_BG_LIGHT,
    RECIDIVIZ_DARK_TEAL,
    RECIDIVIZ_TEXT_DARK,
)
from cj_data_quality.visualization._style import apply_style as _apply_style


def plot_coverage_heatmap(
    matrix: pd.DataFrame,
    title: str = "Cross-State Coverage Matrix",
    figsize: tuple[int, int] = (14, 10),
    cmap: str = "YlGnBu",
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".0%",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap of state x metric completeness.

    Args:
        matrix: DataFrame with states as index, metrics as columns,
                values 0.0-1.0 (completeness).
    """
    _apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]
        assert fig is not None

    if matrix.empty:
        ax.text(0.5, 0.5, "No coverage data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
        fig.tight_layout()
        return fig

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Completeness", "shrink": 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK, pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")  # type: ignore[attr-defined]
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_equity_heatmap(
    equity_df: pd.DataFrame,
    state_col: str = "state_code",
    field_col: str = "field_name",
    value_col: str = "completeness",
    title: str = "Demographic Data Completeness by State",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Heatmap of demographic field completeness across states.

    Args:
        equity_df: DataFrame with state_code, field_name, completeness columns.
    """
    _apply_style()

    if equity_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No equity data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
        fig.tight_layout()
        return fig

    pivot = equity_df.pivot_table(
        index=state_col, columns=field_col, values=value_col, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Completeness"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK, pad=12)
    ax.set_ylabel("State")
    ax.set_xlabel("Demographic Field")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")  # type: ignore[attr-defined]
    fig.tight_layout()
    return fig


def plot_quality_heatmap(
    quality_df: pd.DataFrame,
    entity_col: str = "entity",
    dimension_col: str = "dimension",
    score_col: str = "score",
    title: str = "Data Quality Scores",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Heatmap of quality scores across entities and dimensions.

    Args:
        quality_df: DataFrame with entity, dimension, score columns.
    """
    _apply_style()

    if quality_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No quality data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=RECIDIVIZ_TEXT_DARK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK)
        fig.tight_layout()
        return fig

    pivot = quality_df.pivot_table(
        index=entity_col, columns=dimension_col, values=score_col, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Quality Score"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color=RECIDIVIZ_TEXT_DARK, pad=12)
    fig.tight_layout()
    return fig
