# Quickstart

Two entry points: the **counterfactual engine** (Python API) and the **LL 144 audit CLI**.

---

## Counterfactual engine

```python
from relfair.graph import DependencyGraph
from relfair.mechanisms import LearnedMechanisms
from relfair.counterfactual import batch_counterfactuals, detect_constraint_violations

# 1. Define the causal graph
G = DependencyGraph.from_edges([
    ("sex", "relationship"),
    ("sex", "occupation"),
    ("age", "occupation"),
])

# 2. Add hard rules for deterministic transitions
#    from_val means "only fire when the current value IS Husband"
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",      from_val="Husband")
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")

# 3. Fit structural equations from training data
mechs = LearnedMechanisms(G)
mechs.fit(training_df, cat_cols=["sex", "relationship", "occupation"])

# 4. Generate counterfactuals (vectorised — fast on large datasets)
naive    = batch_counterfactuals(test_df, "sex", "Female", G, mechs, naive=True,  seed=42)
relaware = batch_counterfactuals(test_df, "sex", "Female", G, mechs, naive=False, seed=42)

# 5. Detect hard-rule violations (100% recall — no IsolationForest needed)
violated = detect_constraint_violations(naive, G)
print(f"{violated.sum()} off-manifold rows in naive CFs")
```

!!! warning "`from_val` is load-bearing"
    Without it, a rule `when sex=Female → relationship=Wife` overwrites **every** relationship to `Wife`.
    Always add passthrough rules for values that should be preserved (e.g. `Own-child → Own-child`).

---

## LL 144 audit CLI

```bash
pip install "relfair[cli,report]"

relfair audit predictions.csv \
  --outcome hired \
  --sex   sex_column \
  --race  race_column \
  --pdf   report.pdf \
  --json  report.json \
  --employer    "Acme Corp" \
  --aedt        "Resume Ranker v4.2" \
  --auditor     "Archit Rathod" \
  --cover-period "Jan 1 2025 – Dec 31 2025"
```

Produces:

- **Console table** — selection rates, impact ratios, four-fifths flags, bootstrap CIs
- **PDF** — DCWP-compliant report (requires WeasyPrint + GTK3)
- **JSON** — machine-readable metrics, CIs, p-values, and flags

---

## LL 144 metrics Python API

```python
from relfair.metrics import compute_ll144_metrics
from relfair.report   import write_json, write_pdf

result = compute_ll144_metrics(
    df,
    outcome_col="hired",
    outcome_type="binary",        # or "score"
    sex_col="sex",
    race_col="race",
    n_boot=2000,
    bootstrap_seed=42,
    meta={"employer": "Acme Corp", "auditor": "Archit Rathod"},
)

# Inspect results
for group in result.by_race:
    flag = "⚑" if group.four_fifths_flag else "✓"
    print(f"{flag} {group.group}: ratio={group.ratio:.3f}")

for cell in result.intersectional:
    print(f"{cell.sex} × {cell.race}: ratio={cell.ratio:.3f}")

# Export
write_json(result, "report.json")
write_pdf(result,  "report.pdf")
```

---

## ACS / float-coded categoricals

ACS datasets encode categories as floats (e.g. `SEX=1.0`, `OCCP=3.0`, `MAR=5.0`). Pass them explicitly:

```python
mechs.fit(
    acs_df,
    cat_cols=["SEX", "MAR", "OCCP", "RAC1P"],
)
```

Without `cat_cols`, only `object`/`category`-dtype columns are treated as categorical.
