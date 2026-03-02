"""Coverage analysis: state x metric matrices, demographic completeness."""

from cj_data_quality.coverage.coverage_matrix import (
    build_coverage_matrix,
    identify_coverage_gaps,
    summarize_coverage,
)
from cj_data_quality.coverage.equity_coverage import (
    analyze_demographic_completeness,
    compute_equity_disparity_index,
)

__all__ = [
    "analyze_demographic_completeness",
    "build_coverage_matrix",
    "compute_equity_disparity_index",
    "identify_coverage_gaps",
    "summarize_coverage",
]
