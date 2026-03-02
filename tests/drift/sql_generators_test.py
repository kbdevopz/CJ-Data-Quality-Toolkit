"""Tests for drift SQL generators."""

from cj_data_quality.drift.sql_generators import (
    generate_distribution_shift_sql,
    generate_period_comparison_sql,
)


class TestPeriodComparisonSql:
    def test_contains_table(self) -> None:
        sql = generate_period_comparison_sql(
            "proj.ds.tbl", "dt", "val", "2023-01-01", "2023-04-01"
        )
        assert "`proj.ds.tbl`" in sql

    def test_contains_periods(self) -> None:
        sql = generate_period_comparison_sql(
            "t", "dt", "val", "2023-01-01", "2023-07-01"
        )
        assert "2023-01-01" in sql
        assert "2023-07-01" in sql

    def test_uses_approx_quantiles(self) -> None:
        sql = generate_period_comparison_sql("t", "dt", "val", "2023-01-01", "2023-04-01")
        assert "APPROX_QUANTILES" in sql


class TestDistributionShiftSql:
    def test_contains_table(self) -> None:
        sql = generate_distribution_shift_sql("proj.ds.tbl", "dt", "val")
        assert "`proj.ds.tbl`" in sql

    def test_uses_lag(self) -> None:
        sql = generate_distribution_shift_sql("t", "dt", "val")
        assert "LAG(" in sql

    def test_drift_flag(self) -> None:
        sql = generate_distribution_shift_sql("t", "dt", "val")
        assert "DRIFT_DETECTED" in sql
        assert "STABLE" in sql
