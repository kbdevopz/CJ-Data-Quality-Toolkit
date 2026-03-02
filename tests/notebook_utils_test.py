"""Tests for notebook_utils module."""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pandas as pd

from cj_data_quality.notebook_utils import (
    display_quality_score,
    display_table_profile,
    get_style_path,
    setup_notebook,
    style_null_rates,
    style_quality_scores,
)
from cj_data_quality.profiling.table_profiler import profile_table
from cj_data_quality.types import (
    DimensionScore,
    QualityDimension,
    QualityScore,
)


class TestGetStylePath:
    def test_returns_path(self) -> None:
        path = get_style_path()
        assert isinstance(path, Path)

    def test_path_ends_with_mplstyle(self) -> None:
        path = get_style_path()
        assert path.name == "cj_data_quality.mplstyle"

    def test_style_file_exists(self) -> None:
        assert get_style_path().exists()


class TestDisplayTableProfile:
    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        profile = profile_table(df, "test")
        result = display_table_profile(profile)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        df = pd.DataFrame({"val": [10, 20, 30]})
        profile = profile_table(df, "test")
        result = display_table_profile(profile)
        assert "Column" in result.columns
        assert "Type" in result.columns
        assert "Null Rate" in result.columns


class TestDisplayQualityScore:
    def test_returns_dataframe(self) -> None:
        score = QualityScore(
            entity_name="test",
            composite_score=0.85,
            dimension_scores=[
                DimensionScore(dimension=QualityDimension.COMPLETENESS, score=0.9, weight=0.3),
            ],
            grade="B",
        )
        result = display_quality_score(score)
        assert isinstance(result, pd.DataFrame)

    def test_includes_composite_row(self) -> None:
        score = QualityScore(
            entity_name="test",
            composite_score=0.85,
            dimension_scores=[
                DimensionScore(dimension=QualityDimension.COMPLETENESS, score=0.9, weight=0.3),
            ],
            grade="B",
        )
        result = display_quality_score(score)
        assert any("COMPOSITE" in str(v) for v in result["Dimension"])


class TestStyleNullRates:
    def test_returns_styler(self) -> None:
        df = pd.DataFrame({"null_rate": [0.1, 0.3, 0.6]})
        result = style_null_rates(df)
        assert hasattr(result, "to_html")  # Styler objects have to_html

    def test_critical_highlight(self) -> None:
        df = pd.DataFrame({"null_rate": [0.7]})
        styler = style_null_rates(df)
        html = styler.to_html()
        assert "#CC0000" in html  # critical red

    def test_warning_highlight(self) -> None:
        df = pd.DataFrame({"null_rate": [0.3]})
        styler = style_null_rates(df)
        html = styler.to_html()
        assert "#996600" in html  # warning amber

    def test_good_highlight(self) -> None:
        df = pd.DataFrame({"null_rate": [0.05]})
        styler = style_null_rates(df)
        html = styler.to_html()
        assert "#006600" in html  # good green


class TestStyleQualityScores:
    def test_returns_styler(self) -> None:
        df = pd.DataFrame({"score": [0.5, 0.75, 0.95]})
        result = style_quality_scores(df)
        assert hasattr(result, "to_html")

    def test_good_highlight(self) -> None:
        df = pd.DataFrame({"score": [0.95]})
        html = style_quality_scores(df).to_html()
        assert "#006600" in html

    def test_warning_highlight(self) -> None:
        df = pd.DataFrame({"score": [0.75]})
        html = style_quality_scores(df).to_html()
        assert "#996600" in html

    def test_poor_highlight(self) -> None:
        df = pd.DataFrame({"score": [0.3]})
        html = style_quality_scores(df).to_html()
        assert "#CC0000" in html


class TestSetupNotebook:
    def test_runs_without_error(self) -> None:
        setup_notebook()

    def test_sets_float_format(self) -> None:
        setup_notebook()
        fmt = pd.get_option("display.float_format")
        assert fmt is not None
