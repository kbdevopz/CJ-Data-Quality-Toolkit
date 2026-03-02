"""BigQuery SQL generators for data-quality validation checks.

Each function returns a ready-to-execute BigQuery SQL string that
replicates the corresponding Python validation logic in-warehouse.
"""

from __future__ import annotations


def generate_date_ordering_sql(
    table: str,
    earlier_col: str,
    later_col: str,
) -> str:
    """Generate BigQuery SQL to check date ordering.

    Args:
        table: Fully qualified BigQuery table name
            (e.g. ``project.dataset.table``).
        earlier_col: Column expected to hold the earlier date.
        later_col: Column expected to hold the later date.

    Returns:
        A BigQuery SQL string that counts total checked rows and violations.
    """
    return (
        f"SELECT\n"
        f"  '{earlier_col}' AS earlier_field,\n"
        f"  '{later_col}' AS later_field,\n"
        f"  COUNT(*) AS total_checked,\n"
        f"  COUNTIF({earlier_col} > {later_col}) AS violation_count,\n"
        f"  SAFE_DIVIDE(\n"
        f"    COUNTIF({earlier_col} > {later_col}),\n"
        f"    COUNT(*)\n"
        f"  ) AS violation_rate\n"
        f"FROM `{table}`\n"
        f"WHERE {earlier_col} IS NOT NULL\n"
        f"  AND {later_col} IS NOT NULL"
    )


def generate_completeness_sql(
    table: str,
    columns: list[str],
) -> str:
    """Generate BigQuery SQL to compute null rates for each column.

    Args:
        table: Fully qualified BigQuery table name.
        columns: Column names to measure completeness for.

    Returns:
        A BigQuery SQL string that produces one row per column with its null
        count, total count, and null rate.
    """
    union_parts: list[str] = []
    for col in columns:
        part = (
            f"SELECT\n"
            f"  '{col}' AS column_name,\n"
            f"  COUNT(*) AS total_count,\n"
            f"  COUNTIF({col} IS NULL) AS null_count,\n"
            f"  SAFE_DIVIDE(COUNTIF({col} IS NULL), COUNT(*)) AS null_rate\n"
            f"FROM `{table}`"
        )
        union_parts.append(part)

    return "\nUNION ALL\n".join(union_parts)


def generate_duplicate_check_sql(
    table: str,
    key_columns: list[str],
) -> str:
    """Generate BigQuery SQL to find duplicate rows by key columns.

    Args:
        table: Fully qualified BigQuery table name.
        key_columns: Columns that form the natural key.

    Returns:
        A BigQuery SQL string that counts total rows, distinct keys,
        and the duplicate rate.
    """
    key_list = ", ".join(key_columns)
    return (
        f"SELECT\n"
        f"  COUNT(*) AS total_rows,\n"
        f"  COUNT(DISTINCT STRUCT({key_list})) AS distinct_keys,\n"
        f"  COUNT(*) - COUNT(DISTINCT STRUCT({key_list})) AS duplicate_rows,\n"
        f"  SAFE_DIVIDE(\n"
        f"    COUNT(*) - COUNT(DISTINCT STRUCT({key_list})),\n"
        f"    COUNT(*)\n"
        f"  ) AS duplicate_rate\n"
        f"FROM `{table}`"
    )
