"""Tests for coverage SQL generators."""

from cj_data_quality.coverage.sql_generators import (
    generate_cross_state_matrix_sql,
    generate_demographic_completeness_sql,
)


class TestCrossStateMatrixSql:
    def test_contains_table(self) -> None:
        sql = generate_cross_state_matrix_sql("proj.ds.tbl", "state_code", ["pop", "count"])
        assert "`proj.ds.tbl`" in sql

    def test_includes_all_metrics(self) -> None:
        sql = generate_cross_state_matrix_sql("t", "state_code", ["pop", "admissions", "releases"])
        assert "pop_completeness" in sql
        assert "admissions_completeness" in sql
        assert "releases_completeness" in sql

    def test_groups_by_state(self) -> None:
        sql = generate_cross_state_matrix_sql("t", "state_code", ["pop"])
        assert "GROUP BY" in sql
        assert "state_code" in sql


class TestDemographicCompletenessSql:
    def test_contains_table(self) -> None:
        sql = generate_demographic_completeness_sql("proj.ds.tbl", "state_code", "race")
        assert "`proj.ds.tbl`" in sql

    def test_contains_demo_column(self) -> None:
        sql = generate_demographic_completeness_sql("t", "state_code", "ethnicity")
        assert "ethnicity" in sql

    def test_includes_completeness(self) -> None:
        sql = generate_demographic_completeness_sql("t", "state_code", "race")
        assert "completeness" in sql
        assert "most_common_value" in sql
