"""Generate synthetic criminal justice corrections data.

Real BJS (Bureau of Justice Statistics) data requires manual download from
https://bjs.ojp.gov/. This module generates synthetic data that mimics the
structure and common quality issues found in state-level corrections reporting.

Quality issues are intentionally injected for demonstration and testing of
the CJ Data Quality toolkit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is importable when running as a script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cj_data_quality.constants import US_STATE_CODES  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
_RACE_VALUES = ["WHITE", "BLACK", "HISPANIC", "ASIAN", "AMERICAN_INDIAN", "OTHER"]
_RACE_WEIGHTS = [0.40, 0.33, 0.16, 0.04, 0.03, 0.04]

_ETHNICITY_VALUES = ["HISPANIC", "NOT_HISPANIC"]
_ETHNICITY_WEIGHTS = [0.23, 0.77]

_SEX_VALUES = ["MALE", "FEMALE"]
_SEX_WEIGHTS = [0.93, 0.07]

_AGE_GROUP_BINS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

_DATE_RANGE_START = pd.Timestamp("2015-01-01")
_DATE_RANGE_END = pd.Timestamp("2023-12-31")
_DATE_RANGE_DAYS = (_DATE_RANGE_END - _DATE_RANGE_START).days

# States with deliberately poor data quality (high null rates, missing periods)
_LOW_QUALITY_STATES = [
    "US_AK", "US_WY", "US_MT", "US_SD", "US_ND",
    "US_VT", "US_NH", "US_ME", "US_HI", "US_DE",
]

# States with moderate quality
_MEDIUM_QUALITY_STATES = [
    "US_NM", "US_WV", "US_RI", "US_NE", "US_KS",
    "US_IA", "US_AR", "US_MS", "US_ID", "US_NV",
]


def _assign_age_group(ages: np.ndarray) -> list[str]:
    """Map integer ages to categorical age-group labels."""
    groups: list[str] = []
    for age in ages:
        if age < 25:
            groups.append("18-24")
        elif age < 35:
            groups.append("25-34")
        elif age < 45:
            groups.append("35-44")
        elif age < 55:
            groups.append("45-54")
        elif age < 65:
            groups.append("55-64")
        else:
            groups.append("65+")
    return groups


def _null_rate_for_state(state_code: str) -> float:
    """Return the base null injection rate for a state.

    Low-quality states get 30-50% nulls; medium-quality get 10-25%;
    all others get 2-8%.
    """
    if state_code in _LOW_QUALITY_STATES:
        return 0.30 + 0.20 * (hash(state_code) % 100) / 100.0
    if state_code in _MEDIUM_QUALITY_STATES:
        return 0.10 + 0.15 * (hash(state_code) % 100) / 100.0
    return 0.02 + 0.06 * (hash(state_code) % 100) / 100.0


def _demographic_null_rate(state_code: str) -> float:
    """Return the null rate for demographic fields (race/ethnicity).

    Demographic missingness is generally higher than date missingness.
    """
    base = _null_rate_for_state(state_code)
    # Demographic fields are 1.5x more likely to be missing
    return min(base * 1.5, 0.70)


def generate_synthetic_corrections_data(
    n_records: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic corrections dataset resembling BJS data.

    Parameters
    ----------
    n_records:
        Approximate number of rows to generate.  The actual count may vary
        slightly due to per-state allocation rounding.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: person_id, state_code, admission_date,
        release_date, offense_date, sentence_date, facility_id, race,
        ethnicity, sex, age, age_group, total_population, admission_count,
        release_count, reporting_date.

    Notes
    -----
    Quality issues intentionally injected:
    - Varying null rates per state (5% to 40%+)
    - Date inversions (release before admission) in ~2% of rows
    - Population spikes in random state-quarters
    - Missing reporting periods for some states
    - Missing demographics (race/ethnicity) at varying rates by state
    - A few dates before 1900 and future dates
    """
    rng = np.random.RandomState(seed)
    state_codes = sorted(US_STATE_CODES.keys())
    n_states = len(state_codes)

    # Allocate records roughly proportional to state population hash,
    # but ensure every state gets at least some records.
    raw_weights = np.array(
        [max(300, abs(hash(sc)) % 3000) for sc in state_codes], dtype=float
    )
    raw_weights /= raw_weights.sum()
    per_state_counts = (raw_weights * n_records).astype(int)
    # Ensure minimum of 100 per state
    per_state_counts = np.maximum(per_state_counts, 100)
    actual_total = int(per_state_counts.sum())

    # ----- Pre-allocate arrays -----
    person_ids: list[str] = []
    all_state_codes: list[str] = []
    admission_dates: list[pd.Timestamp | None] = []
    release_dates: list[pd.Timestamp | None] = []
    offense_dates: list[pd.Timestamp | None] = []
    sentence_dates: list[pd.Timestamp | None] = []
    facility_ids: list[str | None] = []
    races: list[str | None] = []
    ethnicities: list[str | None] = []
    sexes: list[str | None] = []
    ages: list[int | None] = []
    age_groups: list[str | None] = []
    total_populations: list[int] = []
    admission_counts: list[int] = []
    release_counts: list[int] = []
    reporting_dates: list[pd.Timestamp | None] = []

    global_idx = 0

    for state_idx, state_code in enumerate(state_codes):
        n_state = int(per_state_counts[state_idx])
        state_rng_seed = seed + state_idx * 137
        state_rng = np.random.RandomState(state_rng_seed)

        null_rate = _null_rate_for_state(state_code)
        demo_null_rate = _demographic_null_rate(state_code)

        # Base population for this state (used for count metrics)
        base_pop = 2000 + abs(hash(state_code)) % 48000

        # Number of facilities in this state
        n_facilities = max(3, abs(hash(state_code + "_fac")) % 25)
        fac_ids = [f"{state_code}_FAC_{i:03d}" for i in range(n_facilities)]

        for i in range(n_state):
            row_idx = global_idx + i
            person_ids.append(f"P{row_idx:07d}")
            all_state_codes.append(state_code)

            # --- Dates ---
            # admission_date: random within the date range
            adm_offset = state_rng.randint(0, _DATE_RANGE_DAYS)
            adm_date = _DATE_RANGE_START + pd.Timedelta(days=int(adm_offset))

            # offense_date: 1-365 days before sentence
            offense_before_sentence_days = state_rng.randint(30, 365)
            # sentence_date: 1-180 days before admission
            sentence_before_adm_days = state_rng.randint(1, 180)
            sent_date = adm_date - pd.Timedelta(days=int(sentence_before_adm_days))
            off_date = sent_date - pd.Timedelta(
                days=int(offense_before_sentence_days)
            )

            # release_date: 30-1095 days after admission
            release_offset = state_rng.randint(30, 1095)
            rel_date = adm_date + pd.Timedelta(days=int(release_offset))

            # reporting_date: first of the quarter containing admission
            rep_date = adm_date - pd.Timedelta(
                days=int((adm_date.month - 1) % 3 * 30 + adm_date.day - 1)
            )
            rep_date = pd.Timestamp(
                year=adm_date.year,
                month=((adm_date.month - 1) // 3) * 3 + 1,
                day=1,
            )

            # --- Inject null dates based on state null rate ---
            if state_rng.random() < null_rate:
                adm_date = None  # type: ignore[assignment]
            if state_rng.random() < null_rate * 0.5:
                rel_date = None  # type: ignore[assignment]
            if state_rng.random() < null_rate * 0.8:
                off_date = None  # type: ignore[assignment]
            if state_rng.random() < null_rate * 0.6:
                sent_date = None  # type: ignore[assignment]

            admission_dates.append(adm_date)
            release_dates.append(rel_date)
            offense_dates.append(off_date)
            sentence_dates.append(sent_date)

            # --- Missing reporting periods for low-quality states ---
            if state_code in _LOW_QUALITY_STATES and state_rng.random() < 0.15:
                reporting_dates.append(None)
            else:
                reporting_dates.append(rep_date)

            # --- Facility ---
            facility_ids.append(state_rng.choice(fac_ids))

            # --- Demographics ---
            if state_rng.random() < demo_null_rate:
                races.append(None)
            else:
                races.append(
                    state_rng.choice(_RACE_VALUES, p=_RACE_WEIGHTS)
                )

            if state_rng.random() < demo_null_rate:
                ethnicities.append(None)
            else:
                ethnicities.append(
                    state_rng.choice(_ETHNICITY_VALUES, p=_ETHNICITY_WEIGHTS)
                )

            if state_rng.random() < demo_null_rate * 0.3:
                sexes.append(None)
            else:
                sexes.append(state_rng.choice(_SEX_VALUES, p=_SEX_WEIGHTS))

            # Age
            age_val = int(state_rng.normal(loc=36, scale=10))
            age_val = max(18, min(age_val, 85))
            if state_rng.random() < demo_null_rate * 0.2:
                ages.append(None)
                age_groups.append(None)
            else:
                ages.append(age_val)
                age_groups.append(_assign_age_group(np.array([age_val]))[0])

            # --- Population / count metrics ---
            quarter_noise = state_rng.normal(0, base_pop * 0.05)
            pop = max(100, int(base_pop + quarter_noise))
            total_populations.append(pop)
            admission_counts.append(max(0, int(pop * 0.02 + state_rng.normal(0, 15))))
            release_counts.append(max(0, int(pop * 0.018 + state_rng.normal(0, 15))))

        global_idx += n_state

    # ----- Build DataFrame -----
    df = pd.DataFrame(
        {
            "person_id": person_ids,
            "state_code": all_state_codes,
            "admission_date": admission_dates,
            "release_date": release_dates,
            "offense_date": offense_dates,
            "sentence_date": sentence_dates,
            "facility_id": facility_ids,
            "race": races,
            "ethnicity": ethnicities,
            "sex": sexes,
            "age": ages,
            "age_group": age_groups,
            "total_population": total_populations,
            "admission_count": admission_counts,
            "release_count": release_counts,
            "reporting_date": reporting_dates,
        }
    )

    # Convert date columns to datetime
    for col in [
        "admission_date",
        "release_date",
        "offense_date",
        "sentence_date",
        "reporting_date",
    ]:
        df[col] = pd.to_datetime(df[col])

    # ===================================================================
    # Inject specific quality issues
    # ===================================================================

    # 1. Date inversions: release_date < admission_date in ~2% of rows
    _inject_date_inversions(df, rng, rate=0.02)

    # 2. Population spikes in random state-quarters
    _inject_population_spikes(df, rng, n_spikes=15)

    # 3. Dates before 1900 (clearly invalid)
    _inject_ancient_dates(df, rng, n_ancient=20)

    # 4. Future dates (past 2025)
    _inject_future_dates(df, rng, n_future=15)

    # Reset index after all mutations
    df = df.reset_index(drop=True)

    return df


def _inject_date_inversions(
    df: pd.DataFrame,
    rng: np.random.RandomState,
    rate: float,
) -> None:
    """Swap release_date and admission_date for a fraction of rows."""
    mask = df["admission_date"].notna() & df["release_date"].notna()
    valid_indices = df.index[mask].tolist()
    n_swap = int(len(valid_indices) * rate)
    swap_indices = rng.choice(valid_indices, size=n_swap, replace=False)

    for idx in swap_indices:
        adm = df.at[idx, "admission_date"]
        rel = df.at[idx, "release_date"]
        # Force release before admission by subtracting time
        df.at[idx, "release_date"] = adm - pd.Timedelta(days=int(rng.randint(1, 60)))
        df.at[idx, "admission_date"] = rel


def _inject_population_spikes(
    df: pd.DataFrame,
    rng: np.random.RandomState,
    n_spikes: int,
) -> None:
    """Multiply total_population by 5-10x for random state-quarter combos."""
    state_codes = df["state_code"].unique()
    spike_states = rng.choice(state_codes, size=min(n_spikes, len(state_codes)), replace=False)

    for state in spike_states:
        state_mask = df["state_code"] == state
        state_indices = df.index[state_mask].tolist()
        if len(state_indices) < 10:
            continue
        # Pick a cluster of consecutive rows to spike
        start = rng.choice(state_indices[:max(1, len(state_indices) - 5)])
        end = min(start + rng.randint(3, 8), df.index[-1])
        multiplier = rng.uniform(5.0, 10.0)
        df.loc[start:end, "total_population"] = (
            df.loc[start:end, "total_population"] * multiplier
        ).astype(int)


def _inject_ancient_dates(
    df: pd.DataFrame,
    rng: np.random.RandomState,
    n_ancient: int,
) -> None:
    """Set a few dates to before 1900 (clearly invalid)."""
    ancient_dates = [
        pd.Timestamp("1850-03-15"),
        pd.Timestamp("1899-12-31"),
        pd.Timestamp("1776-07-04"),
        pd.Timestamp("1800-01-01"),
        pd.Timestamp("1890-06-20"),
    ]
    date_cols = ["admission_date", "release_date", "offense_date", "sentence_date"]

    for _ in range(n_ancient):
        idx = rng.randint(0, len(df))
        col = rng.choice(date_cols)
        df.at[idx, col] = rng.choice(ancient_dates)


def _inject_future_dates(
    df: pd.DataFrame,
    rng: np.random.RandomState,
    n_future: int,
) -> None:
    """Set a few dates to the future (beyond the data range)."""
    future_dates = [
        pd.Timestamp("2028-06-15"),
        pd.Timestamp("2030-01-01"),
        pd.Timestamp("2027-11-30"),
        pd.Timestamp("2035-03-10"),
        pd.Timestamp("2029-07-04"),
    ]
    date_cols = ["admission_date", "release_date", "offense_date", "sentence_date"]

    for _ in range(n_future):
        idx = rng.randint(0, len(df))
        col = rng.choice(date_cols)
        df.at[idx, col] = rng.choice(future_dates)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate synthetic data and save to data/sample/corrections_data.csv."""
    output_dir = Path(__file__).resolve().parent / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "corrections_data.csv"

    print(f"Generating synthetic corrections data (50,000 records, seed=42)...")
    df = generate_synthetic_corrections_data(n_records=50000, seed=42)

    print(f"Writing to {output_path} ...")
    df.to_csv(output_path, index=False)

    print(f"Done. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"States: {df['state_code'].nunique()}")
    print(f"Date range: {df['admission_date'].min()} to {df['admission_date'].max()}")
    print(f"Overall null rate: {df.isnull().mean().mean():.2%}")

    # Print per-state null summary
    print("\nPer-state null rates (top 10 worst):")
    state_nulls = df.groupby("state_code").apply(
        lambda g: g.isnull().mean().mean(), include_groups=False
    ).sort_values(ascending=False)
    for sc, rate in state_nulls.head(10).items():
        print(f"  {sc}: {rate:.2%}")


if __name__ == "__main__":
    main()
