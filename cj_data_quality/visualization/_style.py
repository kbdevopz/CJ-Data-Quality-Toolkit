"""Shared matplotlib style helpers for visualization modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).parent / "cj_data_quality.mplstyle"


def apply_style() -> None:
    """Apply Recidiviz matplotlib style if available."""
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
