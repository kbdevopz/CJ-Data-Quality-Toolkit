"""Tests for cj_data_quality.profiling.type_inference."""

import pandas as pd

from cj_data_quality.profiling.type_inference import infer_column_type
from cj_data_quality.types import ColumnDataType


class TestInferColumnType:
    """Tests for infer_column_type() -- one test per decision-tree path."""

    def test_identifier_by_name(self) -> None:
        series = pd.Series(["P001", "P002", "P003"])
        assert infer_column_type(series, "person_id") == ColumnDataType.IDENTIFIER

    def test_identifier_state_code(self) -> None:
        series = pd.Series(["US_CA", "US_TX", "US_NY"])
        assert infer_column_type(series, "state_code") == ColumnDataType.IDENTIFIER

    def test_identifier_facility_id(self) -> None:
        series = pd.Series(["FAC_A", "FAC_B"])
        assert infer_column_type(series, "facility_id") == ColumnDataType.IDENTIFIER

    def test_temporal_by_name(self) -> None:
        # Even if the dtype is object, the name matches CJ_DATE_FIELDS
        series = pd.Series(["2020-01-01", "2020-06-15"])
        assert infer_column_type(series, "admission_date") == ColumnDataType.TEMPORAL

    def test_temporal_by_dtype(self) -> None:
        series = pd.to_datetime(pd.Series(["2020-01-01", "2020-06-15"]))
        assert infer_column_type(series, "some_unknown_date") == ColumnDataType.TEMPORAL

    def test_temporal_release_date_name(self) -> None:
        series = pd.Series(["2020-03-01"])
        assert infer_column_type(series, "release_date") == ColumnDataType.TEMPORAL

    def test_boolean(self) -> None:
        series = pd.Series([True, False, True, False])
        assert infer_column_type(series, "is_active") == ColumnDataType.BOOLEAN

    def test_numeric_integer(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
        assert infer_column_type(series, "score") == ColumnDataType.NUMERIC

    def test_numeric_float(self) -> None:
        series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5] * 20)
        assert infer_column_type(series, "weight") == ColumnDataType.NUMERIC

    def test_categorical_low_distinct_count(self) -> None:
        # Less than 20 distinct values -> CATEGORICAL
        series = pd.Series(["cat", "dog", "bird"] * 100)
        assert infer_column_type(series, "animal") == ColumnDataType.CATEGORICAL

    def test_categorical_low_cardinality_ratio(self) -> None:
        # 5 distinct values among 500 records -> ratio = 0.01 < 0.05
        values = ["A", "B", "C", "D", "E"] * 100
        series = pd.Series(values)
        assert infer_column_type(series, "category") == ColumnDataType.CATEGORICAL

    def test_free_text_long_strings(self) -> None:
        # Average string length > 50 with high cardinality
        long_strings = [f"This is a very long text entry number {i} " * 3 for i in range(100)]
        series = pd.Series(long_strings)
        result = infer_column_type(series, "description")
        assert result == ColumnDataType.FREE_TEXT

    def test_unknown_fallback(self) -> None:
        # Moderate cardinality, short strings, not matching any special field
        values = [f"val_{i}" for i in range(50)]
        series = pd.Series(values)
        result = infer_column_type(series, "misc_field")
        assert result == ColumnDataType.UNKNOWN

    def test_identifier_takes_priority_over_categorical(self) -> None:
        # "state_code" has low cardinality but should still be IDENTIFIER
        series = pd.Series(["US_CA"] * 100)
        assert infer_column_type(series, "state_code") == ColumnDataType.IDENTIFIER

    def test_temporal_name_takes_priority_over_categorical(self) -> None:
        # "birth_date" with string dtype but in CJ_DATE_FIELDS
        series = pd.Series(["2000-01-01"] * 50)
        assert infer_column_type(series, "birth_date") == ColumnDataType.TEMPORAL

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        result = infer_column_type(series, "empty_col")
        # Empty series with no special name -> UNKNOWN
        assert result == ColumnDataType.UNKNOWN
