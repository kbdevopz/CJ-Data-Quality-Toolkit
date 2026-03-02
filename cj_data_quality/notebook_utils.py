"""Jupyter notebook display helpers for CJ Data Quality toolkit."""

from pathlib import Path
from typing import Any

import pandas as pd

from cj_data_quality.constants import (
    NULL_RATE_CRITICAL,
    NULL_RATE_WARNING,
    QUALITY_SCORE_GOOD,
    QUALITY_SCORE_WARNING,
    RECIDIVIZ_DARK_TEAL,
)
from cj_data_quality.types import ColumnProfile, QualityScore, TableProfile


def style_null_rates(df: pd.DataFrame, null_rate_col: str = "null_rate") -> Any:
    """Apply conditional formatting to a null rate DataFrame.

    Red for critical (>50%), amber for warning (>20%), green otherwise.
    """

    def _highlight(val: Any) -> str:
        v = float(val)
        if v >= NULL_RATE_CRITICAL:
            return "background-color: #FFD0C7; color: #CC0000"
        if v >= NULL_RATE_WARNING:
            return "background-color: #FFF3D0; color: #996600"
        return "background-color: #D0F0D0; color: #006600"

    return df.style.map(_highlight, subset=[null_rate_col])


def style_quality_scores(df: pd.DataFrame, score_col: str = "score") -> Any:
    """Apply conditional formatting to quality score DataFrames."""

    def _highlight(val: Any) -> str:
        v = float(val)
        if v >= QUALITY_SCORE_GOOD:
            return "background-color: #D0F0D0; color: #006600"
        if v >= QUALITY_SCORE_WARNING:
            return "background-color: #FFF3D0; color: #996600"
        return "background-color: #FFD0C7; color: #CC0000"

    return df.style.map(_highlight, subset=[score_col])


def display_table_profile(profile: TableProfile) -> pd.DataFrame:
    """Convert a TableProfile to a display-friendly DataFrame."""
    rows = []
    for cp in profile.column_profiles:
        row = {
            "Column": cp.column_name,
            "Type": cp.inferred_type.value,
            "Null Rate": f"{cp.null_rate:.1%}",
            "Distinct": cp.distinct_count,
            "Cardinality Ratio": f"{cp.cardinality_ratio:.3f}",
        }
        if cp.numeric_stats:
            row["Mean"] = f"{cp.numeric_stats.mean:.1f}"
            row["Std"] = f"{cp.numeric_stats.std:.1f}"
            row["Min"] = f"{cp.numeric_stats.min_value:.1f}"
            row["Max"] = f"{cp.numeric_stats.max_value:.1f}"
        else:
            row["Mean"] = ""
            row["Std"] = ""
            row["Min"] = ""
            row["Max"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def display_quality_score(score: QualityScore) -> pd.DataFrame:
    """Convert a QualityScore to a display-friendly DataFrame."""
    rows = []
    for ds in score.dimension_scores:
        rows.append(
            {
                "Dimension": ds.dimension.value.title(),
                "Score": f"{ds.score:.2f}",
                "Weight": f"{ds.weight:.0%}",
                "Weighted": f"{ds.score * ds.weight:.3f}",
            }
        )
    rows.append(
        {
            "Dimension": f"COMPOSITE (Grade: {score.grade})",
            "Score": f"{score.composite_score:.2f}",
            "Weight": "100%",
            "Weighted": f"{score.composite_score:.3f}",
        }
    )
    return pd.DataFrame(rows)


def get_style_path() -> Path:
    """Return path to the Recidiviz matplotlib style file."""
    return Path(__file__).parent / "visualization" / "cj_data_quality.mplstyle"


def setup_notebook() -> None:
    """Standard notebook setup: imports, style, display options."""
    import matplotlib.pyplot as plt

    style_path = get_style_path()
    if style_path.exists():
        plt.style.use(str(style_path))

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", "{:.4f}".format)
