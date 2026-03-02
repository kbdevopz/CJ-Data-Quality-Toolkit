"""Tests for validation SQL generators."""

from cj_data_quality.validation.sql_generators import (
    generate_completeness_sql,
    generate_date_ordering_sql,
    generate_duplicate_check_sql,
)


class TestDateOrderingSql:
    def test_contains_table(self) -> None:
        sql = generate_date_ordering_sql("proj.ds.tbl", "offense_date", "admission_date")
        assert "`proj.ds.tbl`" in sql

    def test_contains_columns(self) -> None:
        sql = generate_date_ordering_sql("t", "offense_date", "admission_date")
        assert "offense_date" in sql
        assert "admission_date" in sql

    def test_uses_safe_divide(self) -> None:
        sql = generate_date_ordering_sql("t", "a", "b")
        assert "SAFE_DIVIDE" in sql


class TestCompletenessSql:
    def test_single_column(self) -> None:
        sql = generate_completeness_sql("proj.ds.tbl", ["race"])
        assert "`proj.ds.tbl`" in sql
        assert "race" in sql

    def test_multiple_columns_uses_union(self) -> None:
        sql = generate_completeness_sql("t", ["race", "sex", "age"])
        assert sql.count("UNION ALL") == 2

    def test_computes_null_rate(self) -> None:
        sql = generate_completeness_sql("t", ["race"])
        assert "null_rate" in sql


class TestDuplicateCheckSql:
    def test_contains_table(self) -> None:
        sql = generate_duplicate_check_sql("proj.ds.tbl", ["person_id", "state_code"])
        assert "`proj.ds.tbl`" in sql

    def test_includes_key_columns(self) -> None:
        sql = generate_duplicate_check_sql("t", ["person_id", "state_code"])
        assert "person_id" in sql
        assert "state_code" in sql

    def test_computes_duplicate_rate(self) -> None:
        sql = generate_duplicate_check_sql("t", ["id"])
        assert "duplicate_rate" in sql
