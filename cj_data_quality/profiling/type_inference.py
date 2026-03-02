"""Infer the semantic type of a column using a decision-tree approach.

Uses column name matching against known CJ domain fields, dtype inspection,
and heuristic thresholds (cardinality ratio, average string length) to
classify each column into one of the ``ColumnDataType`` members.
"""

import pandas as pd

from cj_data_quality.constants import CJ_DATE_FIELDS, CJ_IDENTIFIER_FIELDS
from cj_data_quality.types import ColumnDataType

_LOW_CARDINALITY_RATIO: float = 0.05
_LOW_CARDINALITY_DISTINCT: int = 20
_FREE_TEXT_AVG_LENGTH: int = 50


def infer_column_type(
    series: pd.Series,
    column_name: str,
) -> ColumnDataType:
    """Infer the semantic data type of a column.

    Decision tree (evaluated in order):
        1. Column name in ``CJ_IDENTIFIER_FIELDS`` -> IDENTIFIER
        2. Column name in ``CJ_DATE_FIELDS`` **or** dtype is datetime -> TEMPORAL
        3. dtype is boolean -> BOOLEAN
        4. dtype is numeric -> NUMERIC
        5. Cardinality ratio < 0.05 **or** distinct count < 20 -> CATEGORICAL
        6. Average string length > 50 -> FREE_TEXT
        7. Otherwise -> UNKNOWN

    Args:
        series: The pandas Series to inspect.
        column_name: Name of the column (used for domain field matching).

    Returns:
        A ``ColumnDataType`` enum member.
    """
    # 1. Identifier fields
    if column_name in CJ_IDENTIFIER_FIELDS:
        return ColumnDataType.IDENTIFIER

    # 2. Temporal fields (by name or dtype)
    if column_name in CJ_DATE_FIELDS or pd.api.types.is_datetime64_any_dtype(series):
        return ColumnDataType.TEMPORAL

    # 3. Boolean
    if pd.api.types.is_bool_dtype(series):
        return ColumnDataType.BOOLEAN

    # 4. Numeric
    if pd.api.types.is_numeric_dtype(series):
        return ColumnDataType.NUMERIC

    # 5. Categorical (low cardinality)
    non_null: pd.Series = series.dropna()
    non_null_count: int = len(non_null)
    distinct_count: int = int(non_null.nunique())

    if non_null_count > 0:
        cardinality_ratio: float = distinct_count / non_null_count
        if cardinality_ratio < _LOW_CARDINALITY_RATIO or distinct_count < _LOW_CARDINALITY_DISTINCT:
            return ColumnDataType.CATEGORICAL

    # 6. Free text (long average string length)
    if non_null_count > 0:
        try:
            avg_len: float = non_null.astype(str).str.len().mean()
            if avg_len > _FREE_TEXT_AVG_LENGTH:
                return ColumnDataType.FREE_TEXT
        except (TypeError, ValueError):
            pass

    # 7. Fallback
    return ColumnDataType.UNKNOWN
