"""Streamlit dashboard for the CJ Data Quality Toolkit.

Interactive explorer for profiling, coverage, equity, drift, anomaly detection,
and quality scoring — powered by the ``cj_data_quality`` library.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is importable when running via ``streamlit run app.py``
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cj_data_quality.constants import (
    CJ_DEMOGRAPHIC_FIELDS,
    CJ_POPULATION_METRICS,
    RECIDIVIZ_COLORS,
    RECIDIVIZ_DARK_TEAL,
    US_STATE_CODES,
)
from cj_data_quality.coverage.coverage_matrix import (
    build_coverage_matrix,
    identify_coverage_gaps,
    summarize_coverage,
)
from cj_data_quality.coverage.equity_coverage import (
    analyze_demographic_completeness,
    compute_equity_disparity_index,
)
from cj_data_quality.drift.distribution_drift import detect_categorical_drift
from cj_data_quality.drift.temporal_drift import (
    detect_temporal_drift,
    summarize_drift_over_time,
)
from cj_data_quality.anomaly.time_series_detector import (
    detect_iqr_anomalies,
    detect_missing_periods,
    detect_rolling_anomalies,
    detect_zscore_anomalies,
)
from cj_data_quality.anomaly.spike_detector import detect_population_anomalies
from cj_data_quality.notebook_utils import display_table_profile, display_quality_score
from cj_data_quality.profiling.table_profiler import profile_table
from cj_data_quality.sample_data import generate_and_load
from cj_data_quality.types import DimensionScore, QualityDimension, QualityScore
from cj_data_quality.validation.completeness_scorer import compute_composite_score
from cj_data_quality.visualization.heatmaps import (
    plot_coverage_heatmap,
    plot_equity_heatmap,
    plot_quality_heatmap,
)
from cj_data_quality.visualization.plots import (
    plot_anomaly_scatter,
    plot_drift_timeline,
    plot_null_rate_bars,
    plot_profile_summary,
    plot_quality_scorecard,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CJ Data Quality Toolkit",
    page_icon=":bar_chart:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Recidiviz palette accents
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        font-family: 'Inter', sans-serif;
    }}
    .severity-critical {{
        background-color: #FFD0C7; color: #CC0000;
        padding: 2px 8px; border-radius: 4px; font-weight: 600;
    }}
    .severity-high {{
        background-color: #FFE0D0; color: #CC4400;
        padding: 2px 8px; border-radius: 4px; font-weight: 600;
    }}
    .severity-medium {{
        background-color: #FFF3D0; color: #996600;
        padding: 2px 8px; border-radius: 4px; font-weight: 600;
    }}
    .severity-low {{
        background-color: #E0F0FF; color: #004C6D;
        padding: 2px 8px; border-radius: 4px; font-weight: 600;
    }}
    .severity-none {{
        background-color: #D0F0D0; color: #006600;
        padding: 2px 8px; border-radius: 4px; font-weight: 600;
    }}
    .grade-a {{ color: #25B894; font-size: 2rem; font-weight: bold; }}
    .grade-b {{ color: #00A5CF; font-size: 2rem; font-weight: bold; }}
    .grade-c {{ color: #FFB84D; font-size: 2rem; font-weight: bold; }}
    .grade-d {{ color: #FF6B4D; font-size: 2rem; font-weight: bold; }}
    .grade-f {{ color: #C44D97; font-size: 2rem; font-weight: bold; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Disclaimer banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="
        background: rgba(255, 184, 77, 0.12);
        border: 1px solid rgba(255, 184, 77, 0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    ">
        <span style="font-size: 1.2rem;">&#9888;&#65039;</span>
        <span style="font-size: 0.85rem; color: #FFB84D;">
            <strong>Disclaimer:</strong> All data shown in this dashboard is
            <strong>100% synthetic / mock data</strong> generated for demonstration
            purposes only. No real criminal justice records are used. This project
            showcases the capabilities of the
            <code>cj_data_quality</code> Python library.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: render matplotlib figure and close it
# ---------------------------------------------------------------------------
def _show_fig(fig: plt.Figure) -> None:
    """Render a matplotlib figure in Streamlit and close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: aggregate to monthly state-level
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Aggregating monthly data...")
def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw records to monthly state-level summaries."""
    work = df.copy()
    work["reporting_date"] = pd.to_datetime(work["reporting_date"], errors="coerce")
    work = work.dropna(subset=["reporting_date"])
    work["month"] = work["reporting_date"].dt.to_period("M").dt.to_timestamp()

    numeric_cols = ["total_population", "admission_count", "release_count"]
    available = [c for c in numeric_cols if c in work.columns]

    agg_dict: dict[str, str] = {c: "mean" for c in available}
    agg_dict["person_id"] = "count"

    monthly = (
        work.groupby(["state_code", "month"])
        .agg(agg_dict)
        .rename(columns={"person_id": "record_count"})
        .reset_index()
    )
    return monthly


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating synthetic data...")
def _load_data(n_records: int) -> pd.DataFrame:
    """Generate and load corrections data with caching."""
    # Delete cached CSV so regeneration uses the requested size
    csv_path = _PROJECT_ROOT / "data" / "sample" / "corrections_data.csv"
    if csv_path.exists():
        csv_path.unlink()
    return generate_and_load(n_records=n_records, seed=42)


@st.cache_data(show_spinner="Profiling table...")
def _cached_profile(df_hash: int, _df: pd.DataFrame) -> object:
    """Cache the table profile (keyed on a hash of the dataframe shape)."""
    return profile_table(_df, table_name="corrections_data")


@st.cache_data(show_spinner="Building coverage matrix...")
def _cached_coverage_matrix(df_hash: int, _df: pd.DataFrame, coverage_cols: tuple[str, ...]) -> pd.DataFrame:
    """Cache the coverage matrix computation."""
    return build_coverage_matrix(_df, "state_code", list(coverage_cols))


@st.cache_data(show_spinner="Analyzing demographic equity...")
def _cached_equity_analysis(df_hash: int, _df: pd.DataFrame) -> list[dict]:
    """Cache the equity analysis results as dicts (serializable)."""
    results = analyze_demographic_completeness(_df, "state_code")
    return [
        {
            "state_code": e.state_code,
            "field_name": e.field_name,
            "completeness": e.completeness,
            "distinct_values": e.distinct_values,
            "most_common_value": e.most_common_value or "",
            "most_common_rate": e.most_common_rate or 0.0,
        }
        for e in results
    ]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {RECIDIVIZ_COLORS[1]};
            padding: 0.5rem 0 0.5rem 0.75rem;
            margin-bottom: 0.25rem;
        ">
            <span style="
                font-size: 1.5rem;
                font-weight: 800;
                letter-spacing: 0.01em;
                color: {RECIDIVIZ_COLORS[1]};
                line-height: 1.25;
            ">CJ Data Quality</span>
            <br/>
            <span style="
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                color: {RECIDIVIZ_DARK_TEAL};
            ">TOOLKIT</span>
        </div>
        <p style="
            font-size: 0.78rem;
            color: #6B7280;
            margin-top: 0.25rem;
            line-height: 1.4;
        ">Criminal Justice Data Quality<br/>Profiling &amp; Validation</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Data Controls")
    sample_size = st.slider(
        "Sample size",
        min_value=5_000,
        max_value=50_000,
        value=10_000,
        step=5_000,
        help="Number of synthetic records to generate.",
    )
    if st.button("Regenerate Data", type="primary"):
        _load_data.clear()
        _cached_profile.clear()
        st.rerun()

    # Load data
    df = _load_data(sample_size)
    tp = _cached_profile(hash((len(df), tuple(df.columns))), df)

    st.divider()
    st.subheader("Summary")
    st.metric("Rows", f"{tp.row_count:,}")
    st.metric("Columns", tp.column_count)
    st.metric("States", df["state_code"].nunique())
    st.metric("Overall Null Rate", f"{tp.overall_null_rate:.1%}")

    st.divider()
    with st.expander("About"):
        st.markdown(
            """
            **CJ Data Quality Toolkit** is a Python library for profiling,
            validating, and scoring criminal justice datasets.

            Built as a companion to the [Recidiviz](https://www.recidiviz.org/)
            data platform, it provides:
            - Column & table profiling with type inference
            - Cross-state coverage analysis
            - Demographic equity audits
            - Distribution drift detection (KS & chi-squared)
            - Anomaly detection (Z-score, IQR, rolling window)
            - Composite quality scoring (5 dimensions, letter grades)
            """
        )

    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 0;
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
            color: #6B7280;
        ">
            Crafted by 😎 <strong>KB</strong>
            &nbsp;
            <a href="https://github.com/kbdevopz" target="_blank" title="GitHub" style="
                color: #6B7280; text-decoration: none; vertical-align: middle;
            "><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: middle;"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12Z"/></svg></a>
            &nbsp;
            <a href="https://www.linkedin.com/in/karlis-baisden-132251191/" target="_blank" title="LinkedIn" style="
                color: #6B7280; text-decoration: none; vertical-align: middle;
            "><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: middle;"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286ZM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065Zm1.782 13.019H3.555V9h3.564v11.452ZM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003Z"/></svg></a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area: 6 tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview & Profiling",
        "Coverage Matrix",
        "Demographic Equity",
        "Drift Detection",
        "Anomaly Detection",
        "Quality Scoring",
    ]
)

# ===== Tab 1: Overview & Profiling ==========================================
with tab1:
    st.header("Overview & Profiling")

    # Metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Row Count", f"{tp.row_count:,}")
    col2.metric("Duplicate Rate", f"{tp.duplicate_rate:.2%}")
    col3.metric("Overall Null Rate", f"{tp.overall_null_rate:.1%}")

    st.subheader("Null Rate by Column")
    fig_null = plot_null_rate_bars(tp.column_profiles)
    _show_fig(fig_null)

    st.subheader("Profile Summary Dashboard")
    fig_profile = plot_profile_summary(tp)
    _show_fig(fig_profile)

    st.subheader("Type Inference Results")
    profile_df = display_table_profile(tp)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # Numeric stats for numeric columns
    numeric_profiles = [
        p for p in tp.column_profiles if p.numeric_stats is not None
    ]
    if numeric_profiles:
        st.subheader("Numeric Column Statistics")
        num_rows = []
        for p in numeric_profiles:
            ns = p.numeric_stats
            num_rows.append(
                {
                    "Column": p.column_name,
                    "Mean": f"{ns.mean:.2f}",
                    "Median": f"{ns.median:.2f}",
                    "Std": f"{ns.std:.2f}",
                    "Min": f"{ns.min_value:.2f}",
                    "Max": f"{ns.max_value:.2f}",
                    "P25": f"{ns.p25:.2f}",
                    "P75": f"{ns.p75:.2f}",
                    "Skewness": f"{ns.skewness:.2f}" if ns.skewness is not None else "N/A",
                }
            )
        st.dataframe(pd.DataFrame(num_rows), use_container_width=True, hide_index=True)


# ===== Tab 2: Coverage Matrix ===============================================
with tab2:
    st.header("Cross-State Coverage Matrix")

    # Build coverage matrix (cached)
    metric_cols = [c for c in CJ_POPULATION_METRICS if c in df.columns]
    date_cols_for_cov = [
        c
        for c in ["admission_date", "release_date", "offense_date", "sentence_date"]
        if c in df.columns
    ]
    all_coverage_cols = metric_cols + date_cols_for_cov + [
        c for c in CJ_DEMOGRAPHIC_FIELDS if c in df.columns
    ]

    _df_hash = hash((len(df), tuple(df.columns)))
    coverage_matrix = _cached_coverage_matrix(_df_hash, df, tuple(all_coverage_cols))

    # Show worst 20 states
    st.subheader("Coverage Heatmap (20 Worst States)")
    worst_states = coverage_matrix.mean(axis=1).nsmallest(20).index
    worst_matrix = coverage_matrix.loc[worst_states]
    fig_cov = plot_coverage_heatmap(worst_matrix, figsize=(16, 10))
    _show_fig(fig_cov)

    # Coverage gaps table
    st.subheader("Coverage Gaps (below 80%)")
    gaps = identify_coverage_gaps(coverage_matrix, threshold=0.8)
    if gaps:
        gap_rows = [
            {
                "State": g.state_code,
                "Metric": g.metric_name,
                "Completeness": f"{g.completeness:.1%}",
            }
            for g in gaps[:50]
        ]
        st.dataframe(
            pd.DataFrame(gap_rows),
            use_container_width=True,
            hide_index=True,
        )
        if len(gaps) > 50:
            st.caption(f"Showing 50 of {len(gaps)} gaps.")
    else:
        st.success("No coverage gaps below 80% threshold.")

    # Coverage summary bar chart
    st.subheader("Mean Completeness per Metric")
    summary = summarize_coverage(coverage_matrix)
    summary_df = pd.DataFrame(
        {"Metric": list(summary.keys()), "Completeness": list(summary.values())}
    ).sort_values("Completeness")

    fig_cov_bar, ax_cov_bar = plt.subplots(figsize=(10, 5))
    colors = [
        "#25B894" if v >= 0.85 else "#FFB84D" if v >= 0.60 else "#FF6B4D"
        for v in summary_df["Completeness"]
    ]
    ax_cov_bar.barh(summary_df["Metric"], summary_df["Completeness"], color=colors)
    ax_cov_bar.set_xlim(0, 1.0)
    ax_cov_bar.set_xlabel("Mean Completeness")
    ax_cov_bar.set_title(
        "Coverage Summary by Metric", fontsize=14, fontweight="bold"
    )
    fig_cov_bar.tight_layout()
    _show_fig(fig_cov_bar)


# ===== Tab 3: Demographic Equity ============================================
with tab3:
    st.header("Demographic Equity Analysis")

    # Equity analysis (cached)
    _df_hash_eq = hash((len(df), tuple(df.columns)))
    equity_rows = _cached_equity_analysis(_df_hash_eq, df)
    equity_df = pd.DataFrame(equity_rows)

    # Equity heatmap — 20 worst states
    st.subheader("Demographic Completeness Heatmap (20 Worst States)")
    state_mean_equity = equity_df.groupby("state_code")["completeness"].mean()
    worst_equity_states = state_mean_equity.nsmallest(20).index.tolist()
    worst_equity_df = equity_df[equity_df["state_code"].isin(worst_equity_states)]

    fig_eq = plot_equity_heatmap(worst_equity_df, figsize=(12, 8))
    _show_fig(fig_eq)

    # Disparity index
    st.subheader("Disparity Index (Top 15 States)")
    if "age" in df.columns and "race" in df.columns:
        disparity = compute_equity_disparity_index(df, "state_code", "race", "age")
        disparity_sorted = sorted(disparity.items(), key=lambda x: x[1], reverse=True)[
            :15
        ]
        if disparity_sorted:
            disp_df = pd.DataFrame(
                disparity_sorted, columns=["State", "Disparity Index"]
            )
            fig_disp, ax_disp = plt.subplots(figsize=(10, 6))
            ax_disp.barh(
                disp_df["State"],
                disp_df["Disparity Index"],
                color=RECIDIVIZ_COLORS[5],
            )
            ax_disp.set_xlabel("Disparity Index (CV)")
            ax_disp.set_title(
                "Age Disparity Across Race Groups by State",
                fontsize=14,
                fontweight="bold",
            )
            ax_disp.invert_yaxis()
            fig_disp.tight_layout()
            _show_fig(fig_disp)
    else:
        st.info("Age and race columns required for disparity analysis.")

    # Worst state-field pairs table
    st.subheader("Worst 20 State-Field Pairs by Completeness")
    worst_pairs = equity_df.nsmallest(20, "completeness").copy()
    worst_pairs_display = pd.DataFrame(
        {
            "State": worst_pairs["state_code"].values,
            "Field": worst_pairs["field_name"].values,
            "Completeness": [f"{v:.1%}" for v in worst_pairs["completeness"]],
            "Distinct Values": worst_pairs["distinct_values"].values,
            "Most Common": worst_pairs["most_common_value"].values,
            "Most Common Rate": [f"{v:.1%}" for v in worst_pairs["most_common_rate"]],
        }
    )
    st.dataframe(worst_pairs_display, use_container_width=True, hide_index=True)


# ===== Tab 4: Drift Detection ===============================================
with tab4:
    st.header("Distribution Drift Detection")

    monthly = _aggregate_monthly(df)

    available_states = sorted(monthly["state_code"].unique())
    if not available_states:
        st.warning("No state-level monthly data available for drift detection.")
        st.stop()

    col_left, col_right = st.columns(2)
    default_state = "US_CA" if "US_CA" in available_states else available_states[0]
    drift_state = col_left.selectbox(
        "State", available_states, index=available_states.index(default_state), key="drift_state"
    )

    value_cols = [
        c
        for c in ["total_population", "admission_count", "release_count", "record_count"]
        if c in monthly.columns
    ]
    drift_value_col = col_right.selectbox(
        "Value column", value_cols, key="drift_value_col"
    )

    # Filter to selected state
    state_monthly = (
        monthly[monthly["state_code"] == drift_state]
        .sort_values("month")
        .reset_index(drop=True)
    )

    if len(state_monthly) >= 4:
        try:
            drift_results = detect_temporal_drift(
                state_monthly,
                date_col="month",
                value_col=drift_value_col,
                period="Q",
            )
        except Exception as exc:
            drift_results = []
            st.error(f"Drift detection failed: {exc}")

        if drift_results:
            # Timeline chart
            st.subheader(f"KS Drift Timeline — {drift_state} / {drift_value_col}")
            drift_summary = summarize_drift_over_time(drift_results)
            drift_summary["period_pair"] = (
                drift_summary["reference_period"]
                + " → "
                + drift_summary["comparison_period"]
            )
            fig_drift = plot_drift_timeline(
                drift_summary,
                title=f"Drift: {drift_state} — {drift_value_col}",
            )
            _show_fig(fig_drift)

            # Results table with severity color coding
            st.subheader("Drift Results")
            display_drift = drift_summary[
                [
                    "reference_period",
                    "comparison_period",
                    "statistic",
                    "p_value",
                    "severity",
                ]
            ].copy()
            display_drift["statistic"] = display_drift["statistic"].map("{:.4f}".format)
            display_drift["p_value"] = display_drift["p_value"].map("{:.6f}".format)
            st.dataframe(display_drift, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough period pairs for drift analysis.")
    else:
        st.warning(
            f"Insufficient monthly data for {drift_state} "
            f"({len(state_monthly)} months). Need at least 4."
        )

    # Categorical drift section
    st.divider()
    st.subheader("Categorical Drift: Race Distribution")
    if "race" in df.columns and "reporting_date" in df.columns:
        state_df = df[df["state_code"] == drift_state].copy()
        state_df["reporting_date"] = pd.to_datetime(
            state_df["reporting_date"], errors="coerce"
        )
        state_df = state_df.dropna(subset=["reporting_date"])

        if len(state_df) > 0:
            state_df["quarter"] = state_df["reporting_date"].dt.to_period("Q")
            quarters = sorted(state_df["quarter"].unique())

            if len(quarters) >= 2:
                first_q = state_df[state_df["quarter"] == quarters[0]]["race"].dropna()
                last_q = state_df[state_df["quarter"] == quarters[-1]]["race"].dropna()

                if len(first_q) == 0 or len(last_q) == 0:
                    st.info("Insufficient non-null race data in first/last quarter.")
                else:
                    try:
                        cat_result = detect_categorical_drift(
                            first_q,
                            last_q,
                            "race",
                            str(quarters[0]),
                            str(quarters[-1]),
                        )

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Chi-squared Statistic", f"{cat_result.statistic:.2f}")
                        c2.metric("p-value", f"{cat_result.p_value:.6f}")
                        c3.metric("Severity", cat_result.severity.value.upper())

                        # Show race distributions side-by-side
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"First quarter: {quarters[0]}")
                            first_vc = first_q.value_counts(normalize=True).reset_index()
                            first_vc.columns = ["Race", "Rate"]
                            st.dataframe(first_vc, hide_index=True)
                        with col_b:
                            st.caption(f"Last quarter: {quarters[-1]}")
                            last_vc = last_q.value_counts(normalize=True).reset_index()
                            last_vc.columns = ["Race", "Rate"]
                            st.dataframe(last_vc, hide_index=True)
                    except Exception as exc:
                        st.error(f"Categorical drift analysis failed: {exc}")
            else:
                st.info("Need at least 2 quarters for categorical drift.")
        else:
            st.info(f"No reporting data for {drift_state}.")
    else:
        st.info("Race and reporting_date columns required.")


# ===== Tab 5: Anomaly Detection =============================================
with tab5:
    st.header("Anomaly Detection")

    col_method, col_threshold, col_state = st.columns(3)
    method = col_method.selectbox(
        "Method",
        ["Z-score", "IQR", "Rolling Window"],
        key="anomaly_method",
    )

    if method == "Z-score":
        threshold = col_threshold.slider(
            "Z-score threshold", 1.5, 5.0, 3.0, 0.5, key="zscore_thresh"
        )
    elif method == "IQR":
        threshold = col_threshold.slider(
            "IQR multiplier", 1.0, 3.0, 1.5, 0.25, key="iqr_thresh"
        )
    else:
        threshold = col_threshold.slider(
            "Num std deviations", 1.0, 4.0, 2.0, 0.5, key="rolling_thresh"
        )

    monthly_anom = _aggregate_monthly(df)  # cached — reuses Tab 4 result
    anom_states = sorted(monthly_anom["state_code"].unique())
    if not anom_states:
        st.warning("No state-level monthly data available for anomaly detection.")
        st.stop()
    default_anom = "US_CA" if "US_CA" in anom_states else anom_states[0]
    anom_state = col_state.selectbox(
        "State", anom_states, index=anom_states.index(default_anom), key="anom_state"
    )

    state_anom_df = (
        monthly_anom[monthly_anom["state_code"] == anom_state]
        .sort_values("month")
        .reset_index(drop=True)
    )

    anom_value_col = "total_population" if "total_population" in state_anom_df.columns else "record_count"

    if len(state_anom_df) >= 3:
        try:
            if method == "Z-score":
                anomalies = detect_zscore_anomalies(
                    state_anom_df,
                    "month",
                    anom_value_col,
                    threshold=threshold,
                    metric_name=anom_value_col,
                    state_code=anom_state,
                )
            elif method == "IQR":
                anomalies = detect_iqr_anomalies(
                    state_anom_df,
                    "month",
                    anom_value_col,
                    multiplier=threshold,
                    metric_name=anom_value_col,
                    state_code=anom_state,
                )
            else:
                window = min(12, len(state_anom_df) - 1)
                anomalies = detect_rolling_anomalies(
                    state_anom_df,
                    "month",
                    anom_value_col,
                    window=max(3, window),
                    num_std=threshold,
                    metric_name=anom_value_col,
                    state_code=anom_state,
                )

            anomaly_indices = []
            for a in anomalies:
                ts = pd.Timestamp(a.timestamp)
                matches = state_anom_df.index[state_anom_df["month"] == ts].tolist()
                anomaly_indices.extend(matches)

            st.subheader(
                f"Anomaly Scatter — {anom_state} / {anom_value_col} ({method})"
            )
            fig_anom = plot_anomaly_scatter(
                state_anom_df,
                "month",
                anom_value_col,
                anomaly_indices,
                title=f"{method} Anomalies: {anom_state}",
            )
            _show_fig(fig_anom)

            # Anomaly results table
            if anomalies:
                st.subheader(f"Detected Anomalies ({len(anomalies)})")
                anom_rows = [
                    {
                        "Date": str(a.timestamp),
                        "Observed": f"{a.observed_value:.1f}",
                        "Expected": f"{a.expected_value:.1f}",
                        "Deviation": f"{a.deviation:.2f}",
                        "Type": a.anomaly_type.value,
                    }
                    for a in anomalies
                ]
                st.dataframe(
                    pd.DataFrame(anom_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("No anomalies detected with current settings.")
        except Exception as exc:
            st.error(f"Anomaly detection failed: {exc}")
    else:
        st.warning(f"Insufficient data for {anom_state} ({len(state_anom_df)} months).")

    # Spike detection: cross-state population anomalies
    st.divider()
    st.subheader("Cross-State Population Anomalies")
    if "total_population" in monthly_anom.columns:
        try:
            pop_anomalies = detect_population_anomalies(
                monthly_anom,
                "month",
                "total_population",
                "state_code",
                threshold=3.0,
            )
        except Exception as exc:
            pop_anomalies = []
            st.error(f"Population anomaly detection failed: {exc}")
        if pop_anomalies:
            st.metric("Population anomalies found", len(pop_anomalies))
            spike_rows = [
                {
                    "State": a.state_code or "",
                    "Date": str(a.timestamp),
                    "Observed": f"{a.observed_value:.0f}",
                    "Expected (mean)": f"{a.expected_value:.0f}",
                    "Z-score": f"{a.deviation:.2f}",
                }
                for a in pop_anomalies[:30]
            ]
            st.dataframe(
                pd.DataFrame(spike_rows),
                use_container_width=True,
                hide_index=True,
            )
            if len(pop_anomalies) > 30:
                st.caption(f"Showing 30 of {len(pop_anomalies)} anomalies.")
        else:
            st.success("No cross-state population anomalies detected.")
    else:
        st.info("total_population column not available.")

    # Missing period detection
    st.divider()
    st.subheader("Missing Reporting Periods")
    missing = detect_missing_periods(
        state_anom_df,
        "month",
        expected_freq="MS",
        metric_name="reporting",
        state_code=anom_state,
    )
    if missing:
        st.warning(f"{len(missing)} missing monthly periods for {anom_state}")
        missing_dates = [str(m.timestamp) for m in missing[:20]]
        st.write(", ".join(missing_dates))
        if len(missing) > 20:
            st.caption(f"Showing 20 of {len(missing)} missing periods.")
    else:
        st.success(f"No missing reporting periods for {anom_state}.")


# ===== Tab 6: Quality Scoring ===============================================


def _compute_timeliness_relative(
    state_dates: pd.Series,
    global_max: pd.Timestamp,
    global_min: pd.Timestamp,
) -> float:
    """Score timeliness relative to the *dataset's* own date range.

    The built-in scorer measures against today, which always gives 0 for
    synthetic / historical data.  This measures: what fraction of this
    state's records fall within the final 25% of the data range?

    A state that stopped reporting early will score lower than one that
    reported through the end of the range.
    """
    dates = pd.to_datetime(state_dates, errors="coerce").dropna()
    if dates.empty:
        return 0.0
    total_range = (global_max - global_min).days
    window_days = max(90, total_range // 4)
    cutoff = global_max - pd.Timedelta(days=window_days)
    return float((dates >= cutoff).sum()) / len(dates)


def _composite_with_relative_timeliness(
    slice_df: pd.DataFrame,
    entity_name: str,
    global_max: pd.Timestamp,
    global_min: pd.Timestamp,
) -> "QualityScore":
    """Composite score that replaces absolute timeliness with relative.

    Computes 4 dimensions normally via the library, then patches in a
    timeliness score anchored to the dataset's own date range.
    """
    from cj_data_quality.validation.completeness_scorer import (
        score_completeness,
        score_consistency,
        score_validity,
        score_uniqueness,
        assign_grade,
    )
    from cj_data_quality.types import DimensionScore, QualityDimension, QualityScore
    from cj_data_quality.constants import DEFAULT_QUALITY_WEIGHTS

    completeness = score_completeness(slice_df)
    consistency = score_consistency(slice_df)
    validity = score_validity(slice_df)
    uniqueness = score_uniqueness(slice_df)

    timeliness_val = _compute_timeliness_relative(
        slice_df["reporting_date"] if "reporting_date" in slice_df.columns else pd.Series(dtype="datetime64[ns]"),
        global_max,
        global_min,
    )
    timeliness = DimensionScore(
        dimension=QualityDimension.TIMELINESS,
        score=timeliness_val,
        weight=DEFAULT_QUALITY_WEIGHTS["timeliness"],
    )

    dimension_scores = [completeness, consistency, timeliness, validity, uniqueness]
    total_weight = sum(ds.weight for ds in dimension_scores)
    composite = (
        sum(ds.score * ds.weight for ds in dimension_scores) / total_weight
        if total_weight > 0
        else 0.0
    )
    composite = max(0.0, min(1.0, composite))

    return QualityScore(
        entity_name=entity_name,
        composite_score=composite,
        dimension_scores=dimension_scores,
        grade=assign_grade(composite),
    )


@st.cache_data(show_spinner="Scoring all states...")
def _compute_all_state_scores(
    _df_hash: int, _df: pd.DataFrame
) -> list[tuple[str, float, str, float, float, float, float, float]]:
    """Compute quality scores for every state. Returns serializable tuples."""
    dates = pd.to_datetime(_df["reporting_date"], errors="coerce").dropna()
    global_max = dates.max() if not dates.empty else pd.Timestamp.today()
    global_min = dates.min() if not dates.empty else global_max

    all_states = sorted(_df["state_code"].unique())
    rows = []
    for sc in all_states:
        state_slice = _df[_df["state_code"] == sc]
        qs = _composite_with_relative_timeliness(
            state_slice, entity_name=sc, global_max=global_max, global_min=global_min,
        )
        dim_map = {ds.dimension.value: ds.score for ds in qs.dimension_scores}
        rows.append((
            sc,
            qs.composite_score,
            qs.grade,
            dim_map.get("completeness", 0.0),
            dim_map.get("consistency", 0.0),
            dim_map.get("timeliness", 0.0),
            dim_map.get("validity", 0.0),
            dim_map.get("uniqueness", 0.0),
        ))
    return rows


with tab6:
    st.header("Quality Scoring")

    # Overall composite score
    st.subheader("Overall Composite Score")
    _report_dates = pd.to_datetime(df["reporting_date"], errors="coerce").dropna()
    _global_max = _report_dates.max() if not _report_dates.empty else pd.Timestamp.today()
    _global_min = _report_dates.min() if not _report_dates.empty else _global_max

    try:
        overall_score = _composite_with_relative_timeliness(
            df,
            entity_name="corrections_data (full dataset)",
            global_max=_global_max,
            global_min=_global_min,
        )
    except Exception as exc:
        st.error(f"Quality scoring failed: {exc}")
        st.stop()

    fig_score = plot_quality_scorecard(overall_score)
    _show_fig(fig_score)

    # Dimension breakdown table
    st.subheader("Dimension Breakdown")
    score_display = display_quality_score(overall_score)
    st.dataframe(score_display, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Per-state scoring — ALL states
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Per-State Quality Scores (All States)")

    score_rows = _compute_all_state_scores(
        hash((len(df), tuple(df.columns))), df
    )
    scores_df = pd.DataFrame(
        score_rows,
        columns=[
            "State",
            "Composite",
            "Grade",
            "Completeness",
            "Consistency",
            "Timeliness",
            "Validity",
            "Uniqueness",
        ],
    )
    scores_df["State Name"] = scores_df["State"].map(
        lambda s: US_STATE_CODES.get(s, s)
    )

    # Grade distribution bar chart
    st.subheader("Grade Distribution")
    grade_counts: dict[str, int] = {}
    for g in ["A", "B", "C", "D", "F"]:
        grade_counts[g] = int((scores_df["Grade"] == g).sum())

    grade_colors = {
        "A": "#25B894",
        "B": "#00A5CF",
        "C": "#FFB84D",
        "D": "#FF6B4D",
        "F": "#C44D97",
    }
    fig_grades, ax_grades = plt.subplots(figsize=(8, 4))
    bars = ax_grades.bar(
        grade_counts.keys(),
        grade_counts.values(),
        color=[grade_colors[g] for g in grade_counts],
    )
    ax_grades.set_ylabel("Count")
    ax_grades.set_xlabel("Grade")
    ax_grades.set_title(
        f"Quality Grade Distribution ({len(scores_df)} States)",
        fontsize=14,
        fontweight="bold",
    )
    for bar, count in zip(bars, grade_counts.values()):
        if count > 0:
            ax_grades.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
    ax_grades.set_ylim(0, max(grade_counts.values(), default=1) + 2)
    fig_grades.tight_layout()
    _show_fig(fig_grades)

    # Quality heatmap — all states x 5 dimensions
    st.subheader("Quality Heatmap (All States x Dimensions)")
    heatmap_rows = []
    for _, row in scores_df.iterrows():
        for dim in ["Completeness", "Consistency", "Timeliness", "Validity", "Uniqueness"]:
            heatmap_rows.append(
                {"entity": row["State"], "dimension": dim, "score": row[dim]}
            )
    heatmap_df = pd.DataFrame(heatmap_rows)
    fig_qheat = plot_quality_heatmap(
        heatmap_df,
        title="Per-State Quality Scores by Dimension",
        figsize=(12, max(8, len(scores_df) * 0.4)),
    )
    _show_fig(fig_qheat)

    # Sortable summary table
    st.subheader("State Scores Summary")
    display_scores = scores_df[
        ["State", "State Name", "Grade", "Composite",
         "Completeness", "Consistency", "Timeliness", "Validity", "Uniqueness"]
    ].copy()
    display_scores = display_scores.sort_values("Composite", ascending=False)

    # Format numeric columns for display
    for col in ["Composite", "Completeness", "Consistency", "Timeliness", "Validity", "Uniqueness"]:
        display_scores[col] = display_scores[col].map("{:.2f}".format)

    st.dataframe(display_scores, use_container_width=True, hide_index=True, height=600)

    # Expandable per-state detail cards
    st.divider()
    st.subheader("Per-State Detail Cards")
    st.caption("Expand any state to see its full scorecard.")

    all_states_sorted = scores_df.sort_values("Composite", ascending=False)
    _dim_weights = {
        "completeness": 0.25, "consistency": 0.20,
        "timeliness": 0.15, "validity": 0.20, "uniqueness": 0.20,
    }
    for _, row in all_states_sorted.iterrows():
        state_code = row["State"]
        grade = row["Grade"]
        composite = row["Composite"]
        state_name = row["State Name"]
        grade_class = f"grade-{grade.lower()}"

        with st.expander(
            f"{grade} | {state_code} — {state_name} | Composite: {composite:.2f}"
        ):
            # Reconstruct QualityScore from already-cached dimension scores
            s = QualityScore(
                entity_name=f"{state_code} ({state_name})",
                composite_score=composite,
                dimension_scores=[
                    DimensionScore(
                        dimension=QualityDimension(dim),
                        score=row[dim.capitalize()],
                        weight=_dim_weights[dim],
                    )
                    for dim in _dim_weights
                ],
                grade=grade,
            )
            state_count = int((df["state_code"] == state_code).sum())
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(
                    f'<div class="{grade_class}">{grade}</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Composite", f"{composite:.2f}")
                st.metric("Records", f"{state_count:,}")
            with col_b:
                fig_s = plot_quality_scorecard(s, figsize=(8, 3))
                _show_fig(fig_s)

            detail_df = display_quality_score(s)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
