"""Tests for cj_data_quality.profiling.sql_generators."""

from cj_data_quality.profiling.sql_generators import (
    generate_column_stats_sql,
    generate_distribution_sql,
    generate_null_rate_sql,
)


_TEST_TABLE: str = "project.dataset.incarceration_periods"
_TEST_COLUMN: str = "admission_date"


class TestGenerateColumnStatsSql:
    """Tests for generate_column_stats_sql()."""

    def test_contains_select(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert "SELECT" in sql

    def test_contains_count(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert "COUNT" in sql

    def test_contains_table_name(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert _TEST_TABLE in sql

    def test_contains_column_name(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert _TEST_COLUMN in sql

    def test_contains_null_rate(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert "null_rate" in sql

    def test_contains_distinct_count(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert "distinct_count" in sql

    def test_contains_from_clause(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert "FROM" in sql

    def test_contains_min_max(self) -> None:
        sql = generate_column_stats_sql(_TEST_TABLE, _TEST_COLUMN)
        assert "MIN" in sql
        assert "MAX" in sql

    def test_different_table_and_column(self) -> None:
        sql = generate_column_stats_sql("other.dataset.table", "release_date")
        assert "other.dataset.table" in sql
        assert "release_date" in sql


class TestGenerateNullRateSql:
    """Tests for generate_null_rate_sql()."""

    # -- No-columns fallback (template) --

    def test_fallback_contains_select(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert "SELECT" in sql

    def test_fallback_contains_table_name(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert _TEST_TABLE in sql

    def test_fallback_contains_null_rate(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert "null_rate" in sql

    def test_fallback_contains_order_by(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert "ORDER BY" in sql

    def test_fallback_contains_count(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert "COUNT" in sql

    def test_fallback_contains_column_name_reference(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert "column_name" in sql

    def test_fallback_contains_note(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE)
        assert "NOTE" in sql

    # -- With columns (UNPIVOT path) --

    def test_unpivot_contains_unpivot(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE, columns=["col_a", "col_b"])
        assert "UNPIVOT" in sql

    def test_unpivot_contains_group_by(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE, columns=["col_a", "col_b"])
        assert "GROUP BY" in sql

    def test_unpivot_contains_order_by(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE, columns=["col_a"])
        assert "ORDER BY" in sql

    def test_unpivot_contains_column_names(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE, columns=["admission_date", "release_date"])
        assert "`admission_date`" in sql
        assert "`release_date`" in sql

    def test_unpivot_contains_table(self) -> None:
        sql = generate_null_rate_sql(_TEST_TABLE, columns=["col_a"])
        assert _TEST_TABLE in sql


class TestGenerateDistributionSql:
    """Tests for generate_distribution_sql()."""

    def test_contains_select(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "SELECT" in sql

    def test_contains_table_name(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert _TEST_TABLE in sql

    def test_contains_column_name(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "age" in sql

    def test_contains_percentile_keywords(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "APPROX_QUANTILES" in sql

    def test_contains_avg_and_stddev(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "AVG" in sql
        assert "STDDEV" in sql

    def test_contains_percentile_offsets(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "p25" in sql
        assert "p50_median" in sql
        assert "p75" in sql

    def test_contains_from_clause(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "FROM" in sql

    def test_contains_where_not_null(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "age")
        assert "IS NOT NULL" in sql

    def test_different_column(self) -> None:
        sql = generate_distribution_sql(_TEST_TABLE, "total_population")
        assert "total_population" in sql
