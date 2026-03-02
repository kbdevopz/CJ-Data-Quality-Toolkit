"""Tests for sample_data module."""

from pathlib import Path

import pandas as pd
import pytest

from cj_data_quality.sample_data import generate_and_load, get_sample_data_path, load_sample_data


class TestGetSampleDataPath:
    def test_returns_path(self) -> None:
        path = get_sample_data_path()
        assert isinstance(path, Path)

    def test_directory_exists(self) -> None:
        path = get_sample_data_path()
        assert path.parent.exists()


class TestGenerateAndLoad:
    def test_returns_dataframe(self) -> None:
        df = generate_and_load(n_records=500, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_respects_n_records(self) -> None:
        df = generate_and_load(n_records=500, seed=42)
        assert len(df) >= 400  # allows some flexibility from generation

    def test_has_expected_columns(self) -> None:
        df = generate_and_load(n_records=500, seed=42)
        assert "person_id" in df.columns
        assert "state_code" in df.columns
        assert "reporting_date" in df.columns

    def test_deterministic_with_seed(self) -> None:
        df1 = generate_and_load(n_records=500, seed=99)
        df2 = generate_and_load(n_records=500, seed=99)
        assert df1["person_id"].tolist() == df2["person_id"].tolist()


class TestLoadSampleData:
    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Sample data file not found"):
            load_sample_data("nonexistent_file_12345.csv")

    def test_load_existing_file(self) -> None:
        # generate_and_load writes a CSV; load_sample_data should read it
        generate_and_load(n_records=200, seed=42)
        df = load_sample_data("corrections_data.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
