"""Shared utility functions for the anomaly detection module."""

from __future__ import annotations

from datetime import date

import pandas as pd


def to_date(value: object) -> date:
    """Coerce a timestamp-like value to a ``datetime.date``."""
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    ts: pd.Timestamp = pd.Timestamp(value)  # type: ignore[arg-type]
    return ts.date()
