"""Constants used throughout the CJ Data Quality toolkit.

Includes Recidiviz color palette, quality thresholds, criminal justice
field names, and US state/territory codes.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Recidiviz 11-color palette (from their public style guide)
# ---------------------------------------------------------------------------
RECIDIVIZ_COLORS: Final[list[str]] = [
    "#004C6D",  # Dark teal (primary)
    "#00A5CF",  # Bright blue
    "#25B894",  # Green
    "#FFB84D",  # Amber
    "#FF6B4D",  # Coral
    "#C44D97",  # Magenta
    "#7B61FF",  # Purple
    "#4DB8FF",  # Light blue
    "#8FD694",  # Light green
    "#FFD180",  # Light amber
    "#FF9E80",  # Light coral
]

RECIDIVIZ_DARK_TEAL: Final[str] = "#004C6D"
RECIDIVIZ_BG_LIGHT: Final[str] = "#F5F6FA"
RECIDIVIZ_TEXT_DARK: Final[str] = "#1B1B1B"
RECIDIVIZ_GRID_GRAY: Final[str] = "#E0E0E0"

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
NULL_RATE_GOOD: Final[float] = 0.05
NULL_RATE_WARNING: Final[float] = 0.20
NULL_RATE_CRITICAL: Final[float] = 0.50

DRIFT_P_VALUE_WARNING: Final[float] = 0.05
DRIFT_P_VALUE_CRITICAL: Final[float] = 0.001

ZSCORE_THRESHOLD_DEFAULT: Final[float] = 3.0
IQR_MULTIPLIER_DEFAULT: Final[float] = 1.5

QUALITY_SCORE_GOOD: Final[float] = 0.85
QUALITY_SCORE_WARNING: Final[float] = 0.60
QUALITY_SCORE_CRITICAL: Final[float] = 0.40

# ---------------------------------------------------------------------------
# Default quality dimension weights (for composite scoring)
# ---------------------------------------------------------------------------
DEFAULT_QUALITY_WEIGHTS: Final[dict[str, float]] = {
    "completeness": 0.30,
    "consistency": 0.25,
    "timeliness": 0.20,
    "validity": 0.15,
    "uniqueness": 0.10,
}

# ---------------------------------------------------------------------------
# Criminal justice domain field names
# ---------------------------------------------------------------------------
CJ_DATE_FIELDS: Final[list[str]] = [
    "admission_date",
    "release_date",
    "offense_date",
    "sentence_date",
    "booking_date",
    "discharge_date",
    "parole_start_date",
    "parole_end_date",
    "birth_date",
    "reporting_date",
]

CJ_DEMOGRAPHIC_FIELDS: Final[list[str]] = [
    "race",
    "ethnicity",
    "sex",
    "age",
    "age_group",
]

CJ_IDENTIFIER_FIELDS: Final[list[str]] = [
    "person_id",
    "state_code",
    "facility_id",
    "case_id",
    "sentence_id",
    "incarceration_period_id",
    "supervision_period_id",
]

CJ_POPULATION_METRICS: Final[list[str]] = [
    "total_population",
    "admission_count",
    "release_count",
    "incarceration_population",
    "supervision_population",
    "parole_population",
    "probation_population",
    "revocation_count",
]

# ---------------------------------------------------------------------------
# US state FIPS codes (50 states + DC + federal)
# ---------------------------------------------------------------------------
US_STATE_CODES: Final[dict[str, str]] = {
    "US_AL": "Alabama",
    "US_AK": "Alaska",
    "US_AZ": "Arizona",
    "US_AR": "Arkansas",
    "US_CA": "California",
    "US_CO": "Colorado",
    "US_CT": "Connecticut",
    "US_DE": "Delaware",
    "US_DC": "District of Columbia",
    "US_FL": "Florida",
    "US_GA": "Georgia",
    "US_HI": "Hawaii",
    "US_ID": "Idaho",
    "US_IL": "Illinois",
    "US_IN": "Indiana",
    "US_IA": "Iowa",
    "US_KS": "Kansas",
    "US_KY": "Kentucky",
    "US_LA": "Louisiana",
    "US_ME": "Maine",
    "US_MD": "Maryland",
    "US_MA": "Massachusetts",
    "US_MI": "Michigan",
    "US_MN": "Minnesota",
    "US_MS": "Mississippi",
    "US_MO": "Missouri",
    "US_MT": "Montana",
    "US_NE": "Nebraska",
    "US_NV": "Nevada",
    "US_NH": "New Hampshire",
    "US_NJ": "New Jersey",
    "US_NM": "New Mexico",
    "US_NY": "New York",
    "US_NC": "North Carolina",
    "US_ND": "North Dakota",
    "US_OH": "Ohio",
    "US_OK": "Oklahoma",
    "US_OR": "Oregon",
    "US_PA": "Pennsylvania",
    "US_RI": "Rhode Island",
    "US_SC": "South Carolina",
    "US_SD": "South Dakota",
    "US_TN": "Tennessee",
    "US_TX": "Texas",
    "US_UT": "Utah",
    "US_VT": "Vermont",
    "US_VA": "Virginia",
    "US_WA": "Washington",
    "US_WV": "West Virginia",
    "US_WI": "Wisconsin",
    "US_WY": "Wyoming",
    "US_FD": "Federal",
}

# Ordered date field pairs for temporal consistency checks
# Each tuple is (earlier_date_field, later_date_field)
CJ_DATE_ORDERING_RULES: Final[list[tuple[str, str]]] = [
    ("offense_date", "sentence_date"),
    ("sentence_date", "admission_date"),
    ("admission_date", "release_date"),
    ("booking_date", "discharge_date"),
    ("parole_start_date", "parole_end_date"),
    ("birth_date", "offense_date"),
    ("birth_date", "admission_date"),
]

# Earliest reasonable date for CJ data
EARLIEST_REASONABLE_DATE: Final[str] = "1900-01-01"
