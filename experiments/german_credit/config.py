"""
German Credit — causal DAG and experiment configuration.

The German Credit dataset (Statlog, UCI) has 1000 rows and 20 attributes.
We use the version with categorical variables pre-processed as strings.

The key variable for the false-positive story:
  personal_status  — encodes BOTH sex and marital status in a single field:
      "male div/sep"       male, divorced or separated
      "male single"        male, single
      "male mar/wid"       male, married or widowed
      "female div/dep/mar" female, divorced, dependent, or married

  Naive flip: change personal_status to the opposite sex but keep the marital
  sub-category unchanged → incoherent input (no "female single" exists in
  this encoding). The model behaviour on this out-of-distribution input may
  spuriously flip, generating a FALSE POSITIVE.

  Rel-aware flip: apply the hard rule mapping to the nearest coherent female
  equivalent, then let learned mechanisms re-derive remaining descendants.

Hard rule mapping (Male -> Female intervention):
  "male div/sep"  -> "female div/dep/mar"  (both non-single, non-married)
  "male single"   -> "female div/dep/mar"  (closest — only female code exists)
  "male mar/wid"  -> "female div/dep/mar"  (married variant)

Feature reference (UCI German Credit, 20 features):
  Numeric : duration, credit_amount, installment_rate, residence_since,
            age, existing_credits, num_dependents
  Categorical : status, credit_history, purpose, savings, employment,
                personal_status, other_debtors, property, other_plans,
                housing, job, telephone, foreign_worker
"""

TARGET_COL = "target"   # 1 = good credit, 0 = bad credit (inverted from UCI 2=bad)

# Column lists (must match download/preprocess order)
NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]
CATEGORICAL_COLS = [
    "status", "credit_history", "purpose", "savings", "employment",
    "personal_status", "other_debtors", "property", "other_plans",
    "housing", "job", "telephone", "foreign_worker",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Causal DAG edges
# personal_status  -> credit_amount   (married men typically borrow more)
# personal_status  -> duration        (soft: loan duration varies by household)
# age              -> employment      (older workers have longer employment)
# age              -> savings         (older workers have more savings)
EDGES = [
    ("personal_status", "credit_amount"),
    ("personal_status", "duration"),
    ("age", "employment"),
    ("age", "savings"),
]

# No additional hard rules beyond the direct intervention on personal_status.
# personal_status is the root node (no parents in this graph), so its value is
# set directly by the intervention.  credit_amount and duration are re-derived
# from their learned mechanisms given the new personal_status value.
HARD_RULES: list[dict] = []

# Intervention: flip the sex dimension of personal_status
INTERVENTION = {
    "attr": "personal_status",
    "from_vals": ("male div/sep", "male single", "male mar/wid"),
    "to_val": "female div/dep/mar",
    "label": "Male -> Female (personal_status)",
}
