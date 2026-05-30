"""
Adult/Census Income — causal DAG and experiment configuration.

This is the canonical false-positive demonstration for the FairLens paper.

The key structural fact
-----------------------
In the UCI Adult dataset, `relationship` is sex-typed with near-perfect exclusivity:

    relationship=Husband  ->  sex=Male   (12,462 of 12,463 rows)
    relationship=Wife     ->  sex=Female (1,405 of 1,406 rows)

This means:
  - Naive flip (Male -> Female, keep relationship=Husband) creates a combination
    that appears only ONCE in 30,162 rows.  IsolationForest flags essentially
    100% of these naive CFs as off-manifold.

  - Relationship-aware flip applies the hard rule: Husband -> Wife.
    (Female, Wife) is a legitimate, in-distribution combination.

For these off-manifold rows, the model extrapolates to (Female, Husband) —
a point it has never seen — and may return an arbitrary or biased prediction.
Any flip on that extrapolated input is a FALSE POSITIVE.

When we correct the input to (Female, Wife), the model's prediction is
calibrated against real training data.  If the prediction no longer flips,
the original flip was indeed a false positive.

Feature reference
-----------------
  age             — continuous
  workclass       — categorical (Private, Self-emp-not-inc, …)
  fnlwgt          — continuous (survey weight)
  education       — categorical (Bachelors, Masters, …)
  education_num   — continuous (ordinal encoding of education)
  marital_status  — categorical (Married-civ-spouse, Divorced, …)
  occupation      — categorical (Tech-support, Craft-repair, …)
  relationship    — categorical (Husband, Wife, Own-child, …)
  race            — categorical (White, Black, …)
  sex             — categorical (Male, Female)   ← PROTECTED
  capital_gain    — continuous
  capital_loss    — continuous
  hours_per_week  — continuous
  native_country  — categorical
"""

FEATURE_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
]
TARGET_COL = "label"

CONTINUOUS_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]
CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

# ---------------------------------------------------------------------------
# Causal DAG edges
# ---------------------------------------------------------------------------
# sex -> relationship    Husband/Wife are sex-coded (the hard-rule case)
# sex -> marital_status  Marital status rates differ by sex
# sex -> occupation      Occupational segregation
# sex -> hours_per_week  Part-time work skews female
# age -> marital_status  Older -> more likely married/widowed
# age -> occupation      Age cohort affects occupation
# age -> hours_per_week  Age affects work intensity
EDGES = [
    ("sex", "relationship"),
    ("sex", "marital_status"),
    ("sex", "occupation"),
    ("sex", "hours_per_week"),
    ("age", "marital_status"),
    ("age", "occupation"),
    ("age", "hours_per_week"),
]

# ---------------------------------------------------------------------------
# Hard rules — conditional transitions on relationship
#
# Male -> Female intervention:
#   Husband  -> Wife          (the false-positive case we're fixing)
#   Own-child -> Own-child    (gender-neutral — no change)
#   Not-in-family -> same
#   Unmarried -> same
#   Other-relative -> same
#
# Using from_val so the rule is CONDITIONAL on the current relationship value.
# Without from_val, "when sex=Female -> Wife" would turn every relationship
# into Wife, which is wrong for Own-child rows.
# ---------------------------------------------------------------------------
HARD_RULES = [
    # The critical one: Husband -> Wife
    dict(node="relationship", when={"sex": "Female"}, value="Wife",           from_val="Husband"),
    # Passthrough rules — preserve the value for gender-neutral relationships
    dict(node="relationship", when={"sex": "Female"}, value="Own-child",      from_val="Own-child"),
    dict(node="relationship", when={"sex": "Female"}, value="Not-in-family",  from_val="Not-in-family"),
    dict(node="relationship", when={"sex": "Female"}, value="Unmarried",      from_val="Unmarried"),
    dict(node="relationship", when={"sex": "Female"}, value="Other-relative", from_val="Other-relative"),
]

# Intervention: Male -> Female
INTERVENTION = {
    "attr":    "sex",
    "from_val_attr": "Male",
    "to_val":  "Female",
    "label":   "Male -> Female",
}
