"""Shared test fixtures for CJ Data Quality tests.

Provides small DataFrames with known quality issues for deterministic testing.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_incarceration_df() -> pd.DataFrame:
    """A small, clean incarceration DataFrame with no quality issues."""
    np.random.seed(42)
    n = 100
    base_date = date(2020, 1, 1)
    return pd.DataFrame(
        {
            "person_id": [f"P{i:04d}" for i in range(n)],
            "state_code": np.random.choice(
                ["US_CA", "US_TX", "US_NY", "US_FL", "US_OH"], n
            ),
            "admission_date": pd.to_datetime(
                [base_date + timedelta(days=int(d)) for d in np.random.randint(0, 365, n)]
            ),
            "release_date": pd.to_datetime(
                [
                    base_date + timedelta(days=int(d) + int(r))
                    for d, r in zip(
                        np.random.randint(0, 365, n),
                        np.random.randint(30, 730, n),
                    )
                ]
            ),
            "facility_id": np.random.choice(
                ["FAC_A", "FAC_B", "FAC_C", "FAC_D"], n
            ),
            "race": np.random.choice(
                ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"], n
            ),
            "sex": np.random.choice(["MALE", "FEMALE"], n, p=[0.9, 0.1]),
            "age": np.random.randint(18, 70, n),
            "total_population": np.random.randint(500, 5000, n),
        }
    )


@pytest.fixture
def dirty_incarceration_df() -> pd.DataFrame:
    """An incarceration DataFrame with intentional quality issues.

    Issues injected:
    - 20% null admission_date
    - 15% null race
    - 5 rows with release_date < admission_date (date ordering violation)
    - 2 rows with future dates
    - 3 rows with dates before 1900
    - 10 duplicate rows
    - 1 population spike (value 10x normal)
    """
    np.random.seed(99)
    n = 200
    base_date = date(2020, 1, 1)

    admission_dates = [
        base_date + timedelta(days=int(d)) for d in np.random.randint(0, 365, n)
    ]
    release_dates = [
        ad + timedelta(days=int(r))
        for ad, r in zip(admission_dates, np.random.randint(30, 730, n))
    ]

    df = pd.DataFrame(
        {
            "person_id": [f"P{i:04d}" for i in range(n)],
            "state_code": np.random.choice(
                ["US_CA", "US_TX", "US_NY", "US_FL", "US_OH"], n
            ),
            "admission_date": pd.to_datetime(admission_dates),
            "release_date": pd.to_datetime(release_dates),
            "facility_id": np.random.choice(
                ["FAC_A", "FAC_B", "FAC_C", "FAC_D"], n
            ),
            "race": np.random.choice(
                ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"], n
            ),
            "sex": np.random.choice(["MALE", "FEMALE"], n, p=[0.9, 0.1]),
            "age": np.random.randint(18, 70, n),
            "total_population": np.random.randint(500, 5000, n),
        }
    )

    # Inject nulls: 20% admission_date
    null_mask = np.random.choice(n, size=int(n * 0.20), replace=False)
    df.loc[null_mask, "admission_date"] = pd.NaT

    # Inject nulls: 15% race
    null_mask_race = np.random.choice(n, size=int(n * 0.15), replace=False)
    df.loc[null_mask_race, "race"] = None

    # Date ordering violations: release < admission for 5 rows
    for idx in range(5):
        if pd.notna(df.loc[idx, "admission_date"]):
            df.loc[idx, "release_date"] = df.loc[idx, "admission_date"] - timedelta(
                days=10
            )

    # Future dates
    df.loc[10, "admission_date"] = pd.Timestamp("2030-06-15")
    df.loc[11, "release_date"] = pd.Timestamp("2028-12-01")

    # Dates before 1900
    df.loc[15, "admission_date"] = pd.Timestamp("1850-03-01")
    df.loc[16, "admission_date"] = pd.Timestamp("1899-12-31")
    df.loc[17, "release_date"] = pd.Timestamp("1800-01-01")

    # Duplicate rows
    duplicates = df.iloc[:10].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Population spike
    df.loc[50, "total_population"] = 50000  # 10x normal

    return df


@pytest.fixture
def multi_state_population_df() -> pd.DataFrame:
    """Population counts across states and quarters for coverage/drift testing."""
    np.random.seed(123)
    states = ["US_CA", "US_TX", "US_NY", "US_FL", "US_OH"]
    quarters = pd.date_range("2020-01-01", "2023-12-31", freq="QS")

    rows = []
    for state in states:
        base_pop = np.random.randint(5000, 50000)
        for q in quarters:
            pop = int(base_pop + np.random.normal(0, base_pop * 0.05))
            rows.append(
                {
                    "state_code": state,
                    "reporting_date": q,
                    "total_population": pop,
                    "admission_count": int(pop * 0.02 + np.random.normal(0, 20)),
                    "release_count": int(pop * 0.018 + np.random.normal(0, 20)),
                }
            )

    df = pd.DataFrame(rows)

    # Inject missing quarter for TX
    tx_mask = (df["state_code"] == "US_TX") & (
        df["reporting_date"] == pd.Timestamp("2022-04-01")
    )
    df = df[~tx_mask].reset_index(drop=True)

    # Inject spike in CA Q3 2022
    ca_q3_mask = (df["state_code"] == "US_CA") & (
        df["reporting_date"] == pd.Timestamp("2022-07-01")
    )
    df.loc[ca_q3_mask, "total_population"] *= 5

    return df


@pytest.fixture
def demographic_coverage_df() -> pd.DataFrame:
    """DataFrame with varying demographic completeness across states."""
    np.random.seed(456)
    n_per_state = 50
    states = ["US_CA", "US_TX", "US_NY", "US_FL", "US_OH"]

    rows = []
    # CA: full coverage
    for i in range(n_per_state):
        rows.append(
            {
                "state_code": "US_CA",
                "person_id": f"CA_{i:04d}",
                "race": np.random.choice(
                    ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]
                ),
                "ethnicity": np.random.choice(["HISPANIC", "NOT_HISPANIC"]),
                "sex": np.random.choice(["MALE", "FEMALE"]),
                "age": np.random.randint(18, 70),
            }
        )

    # TX: 40% missing race, 60% missing ethnicity
    for i in range(n_per_state):
        rows.append(
            {
                "state_code": "US_TX",
                "person_id": f"TX_{i:04d}",
                "race": (
                    np.random.choice(["WHITE", "BLACK", "HISPANIC"])
                    if np.random.random() > 0.4
                    else None
                ),
                "ethnicity": (
                    np.random.choice(["HISPANIC", "NOT_HISPANIC"])
                    if np.random.random() > 0.6
                    else None
                ),
                "sex": np.random.choice(["MALE", "FEMALE"]),
                "age": np.random.randint(18, 70),
            }
        )

    # NY: full demographics, but only 2 race categories
    for i in range(n_per_state):
        rows.append(
            {
                "state_code": "US_NY",
                "person_id": f"NY_{i:04d}",
                "race": np.random.choice(["WHITE", "BLACK"]),
                "ethnicity": np.random.choice(["HISPANIC", "NOT_HISPANIC"]),
                "sex": np.random.choice(["MALE", "FEMALE"]),
                "age": np.random.randint(18, 70),
            }
        )

    # FL: 80% missing ethnicity
    for i in range(n_per_state):
        rows.append(
            {
                "state_code": "US_FL",
                "person_id": f"FL_{i:04d}",
                "race": np.random.choice(
                    ["WHITE", "BLACK", "HISPANIC", "OTHER"]
                ),
                "ethnicity": (
                    np.random.choice(["HISPANIC", "NOT_HISPANIC"])
                    if np.random.random() > 0.8
                    else None
                ),
                "sex": np.random.choice(["MALE", "FEMALE"]),
                "age": np.random.randint(18, 70),
            }
        )

    # OH: complete data
    for i in range(n_per_state):
        rows.append(
            {
                "state_code": "US_OH",
                "person_id": f"OH_{i:04d}",
                "race": np.random.choice(
                    ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]
                ),
                "ethnicity": np.random.choice(["HISPANIC", "NOT_HISPANIC"]),
                "sex": np.random.choice(["MALE", "FEMALE"]),
                "age": np.random.randint(18, 70),
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def time_series_with_anomalies_df() -> pd.DataFrame:
    """Monthly time series with injected anomalies for detection testing."""
    np.random.seed(789)
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="MS")
    values = 1000 + np.cumsum(np.random.normal(0, 20, len(dates)))

    df = pd.DataFrame({"date": dates, "value": values})

    # Inject anomalies
    df.loc[12, "value"] *= 3.0   # Spike at month 12
    df.loc[30, "value"] *= 0.2   # Drop at month 30
    df.loc[48, "value"] *= 2.5   # Spike at month 48

    return df
