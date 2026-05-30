"""
ACS Income task — causal DAG and experiment configuration.

Feature reference (ACSIncome task):
  AGEP  — age in years                            continuous
  COW   — class of worker (1-9 codes)             categorical
  SCHL  — educational attainment (1-24 codes)     categorical
  MAR   — marital status (1-5 codes)              categorical
  OCCP  — occupation (4-digit SOC codes, 529 unique)  categorical
  POBP  — place of birth (country/state codes)    categorical
  RELP  — relationship to reference person (0-17) categorical
  WKHP  — usual hours worked per week             continuous
  SEX   — 1=Male, 2=Female                        categorical (protected)
  RAC1P — race (1-9 codes)                        categorical (protected)
"""

FEATURE_COLS = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
TARGET_COL = "label"

# Truly continuous (numeric passthrough for mechanisms + classifier)
CONTINUOUS_COLS = ["AGEP", "WKHP"]

# Integer-coded categoricals (need OrdinalEncoder even though dtype is float64)
CATEGORICAL_COLS = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]

# ---------------------------------------------------------------------------
# Causal dependency DAG  (conservative — only edges with strong evidence)
#
# Sex → MAR   : marital status distributions differ significantly by sex
#               (women more often widowed; men more often never-married in
#               some age cohorts; IPUMS documentation confirms dependency)
# Sex → OCCP  : occupational segregation — sex predicts occupation code
# Sex → WKHP  : part-time work is disproportionately female
# Sex → COW   : self-employment and government employment rates differ by sex
# Sex → RELP  : reference person (RELP=0) skews heavily Male; householder
#               vs. spouse role is sex-patterned even in ACS coding
#
# Age → MAR   : strong predictor — older people more likely married/widowed
# Age → SCHL  : educational attainment differs by age cohort (pre/post
#               Title IX, community-college expansion)
# Age → WKHP  : young workers and near-retirement workers work fewer hours
# ---------------------------------------------------------------------------
EDGES = [
    ("SEX", "MAR"),
    ("SEX", "OCCP"),
    ("SEX", "WKHP"),
    ("SEX", "COW"),
    ("SEX", "RELP"),
    ("AGEP", "MAR"),
    ("AGEP", "SCHL"),
    ("AGEP", "WKHP"),
]

# No deterministic hard rules for ACS Income — all dependencies are soft.
# (Unlike Adult/Census where 'relationship: Husband/Wife' is sex-coded.)
HARD_RULES: list[dict] = []

# ---------------------------------------------------------------------------
# Intervention: Male → Female  (SEX: 1.0 → 2.0)
# ---------------------------------------------------------------------------
INTERVENTION = {
    "attr": "SEX",
    "from_val": 1.0,   # Male
    "to_val": 2.0,     # Female
    "label": "Male -> Female",
}
