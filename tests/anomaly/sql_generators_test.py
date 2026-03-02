"""Tests for anomaly SQL generators."""

from cj_data_quality.anomaly.sql_generators import (
    generate_rolling_stats_sql,
    generate_zscore_detection_sql,
)


class TestZscoreDetectionSql:
    def test_contains_table(self) -> None:
        sql = generate_zscore_detection_sql("proj.ds.tbl", "dt", "val", 3.0)
        assert "`proj.ds.tbl`" in sql

    def test_contains_threshold(self) -> None:
        sql = generate_zscore_detection_sql("t", "dt", "val", 2.5)
        assert "2.5" in sql

    def test_contains_columns(self) -> None:
        sql = generate_zscore_detection_sql("t", "report_date", "pop", 3.0)
        assert "report_date" in sql
        assert "pop" in sql

    def test_uses_safe_divide(self) -> None:
        sql = generate_zscore_detection_sql("t", "dt", "val", 3.0)
        assert "SAFE_DIVIDE" in sql


class TestRollingStatsSql:
    def test_contains_table(self) -> None:
        sql = generate_rolling_stats_sql("proj.ds.tbl", "dt", "val", 12)
        assert "`proj.ds.tbl`" in sql

    def test_window_size(self) -> None:
        sql = generate_rolling_stats_sql("t", "dt", "val", 12)
        assert "11 PRECEDING" in sql

    def test_contains_rolling_mean(self) -> None:
        sql = generate_rolling_stats_sql("t", "dt", "val", 6)
        assert "rolling_mean" in sql
        assert "rolling_std" in sql
