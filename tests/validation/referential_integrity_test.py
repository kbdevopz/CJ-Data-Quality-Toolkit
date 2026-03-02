"""Tests for referential integrity validation checks."""

from __future__ import annotations

import pandas as pd
import pytest

from cj_data_quality.validation.referential_integrity import (
    check_cross_table_consistency,
    check_foreign_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parent_df() -> pd.DataFrame:
    """Parent table with a primary key."""
    return pd.DataFrame(
        {
            "facility_id": ["FAC_A", "FAC_B", "FAC_C", "FAC_D"],
            "facility_name": ["Alpha", "Bravo", "Charlie", "Delta"],
        }
    )


@pytest.fixture
def child_df_clean(parent_df: pd.DataFrame) -> pd.DataFrame:
    """Child table where every facility_id exists in the parent."""
    return pd.DataFrame(
        {
            "person_id": ["P001", "P002", "P003", "P004", "P005"],
            "facility_id": ["FAC_A", "FAC_B", "FAC_A", "FAC_C", "FAC_D"],
        }
    )


@pytest.fixture
def child_df_with_orphans() -> pd.DataFrame:
    """Child table with orphan facility_id values."""
    return pd.DataFrame(
        {
            "person_id": [
                "P001",
                "P002",
                "P003",
                "P004",
                "P005",
                "P006",
                "P007",
            ],
            "facility_id": [
                "FAC_A",
                "FAC_B",
                "FAC_X",  # orphan
                "FAC_C",
                "FAC_Y",  # orphan
                "FAC_Z",  # orphan
                None,     # null (should be excluded)
            ],
        }
    )


@pytest.fixture
def table_a() -> pd.DataFrame:
    """Table A for cross-table consistency tests."""
    return pd.DataFrame(
        {"person_id": ["P001", "P002", "P003", "P004", "P005"]}
    )


@pytest.fixture
def table_b() -> pd.DataFrame:
    """Table B for cross-table consistency tests."""
    return pd.DataFrame(
        {"person_id": ["P003", "P004", "P005", "P006", "P007"]}
    )


# ---------------------------------------------------------------------------
# Tests: check_foreign_key
# ---------------------------------------------------------------------------


class TestCheckForeignKey:
    """Tests for check_foreign_key."""

    def test_no_orphans(
        self, child_df_clean: pd.DataFrame, parent_df: pd.DataFrame
    ) -> None:
        """A clean child table should have zero orphans."""
        result = check_foreign_key(
            child_df_clean, "facility_id", parent_df, "facility_id"
        )
        assert result["orphan_count"] == 0
        assert result["orphan_rate"] == 0.0
        assert result["sample_orphans"] == []
        assert result["total_child_values"] == 5

    def test_detects_orphans(
        self, child_df_with_orphans: pd.DataFrame, parent_df: pd.DataFrame
    ) -> None:
        """Orphan values should be detected and counted."""
        result = check_foreign_key(
            child_df_with_orphans, "facility_id", parent_df, "facility_id"
        )
        # 6 non-null child values, 3 are orphans
        assert result["total_child_values"] == 6
        assert result["orphan_count"] == 3
        assert result["orphan_rate"] == pytest.approx(3 / 6)
        assert set(result["sample_orphans"]) == {"FAC_X", "FAC_Y", "FAC_Z"}

    def test_sample_orphans_capped_at_ten(self) -> None:
        """sample_orphans should contain at most 10 values."""
        parent = pd.DataFrame({"id": [1]})
        child = pd.DataFrame({"id": list(range(2, 25))})  # 23 orphans
        result = check_foreign_key(child, "id", parent, "id")
        assert len(result["sample_orphans"]) <= 10

    def test_empty_child_table(self, parent_df: pd.DataFrame) -> None:
        """An empty child table should return zero counts."""
        empty_child = pd.DataFrame({"facility_id": pd.Series([], dtype=str)})
        result = check_foreign_key(
            empty_child, "facility_id", parent_df, "facility_id"
        )
        assert result["total_child_values"] == 0
        assert result["orphan_count"] == 0
        assert result["orphan_rate"] == 0.0


# ---------------------------------------------------------------------------
# Tests: check_cross_table_consistency
# ---------------------------------------------------------------------------


class TestCheckCrossTableConsistency:
    """Tests for check_cross_table_consistency."""

    def test_partial_overlap(
        self, table_a: pd.DataFrame, table_b: pd.DataFrame
    ) -> None:
        """Two partially overlapping tables should be measured correctly."""
        result = check_cross_table_consistency(table_a, table_b, "person_id")
        assert result["a_only_count"] == 2   # P001, P002
        assert result["b_only_count"] == 2   # P006, P007
        assert result["both_count"] == 3     # P003, P004, P005
        assert result["overlap_rate"] == pytest.approx(3 / 7)

    def test_full_overlap(self) -> None:
        """Two identical tables should have 100% overlap."""
        df = pd.DataFrame({"key": [1, 2, 3]})
        result = check_cross_table_consistency(df, df, "key")
        assert result["a_only_count"] == 0
        assert result["b_only_count"] == 0
        assert result["both_count"] == 3
        assert result["overlap_rate"] == 1.0

    def test_no_overlap(self) -> None:
        """Disjoint tables should have 0% overlap."""
        df_a = pd.DataFrame({"key": [1, 2]})
        df_b = pd.DataFrame({"key": [3, 4]})
        result = check_cross_table_consistency(df_a, df_b, "key")
        assert result["both_count"] == 0
        assert result["overlap_rate"] == 0.0

    def test_empty_tables(self) -> None:
        """Two empty tables should return zero counts."""
        df_a = pd.DataFrame({"key": pd.Series([], dtype=int)})
        df_b = pd.DataFrame({"key": pd.Series([], dtype=int)})
        result = check_cross_table_consistency(df_a, df_b, "key")
        assert result["both_count"] == 0
        assert result["overlap_rate"] == 0.0
