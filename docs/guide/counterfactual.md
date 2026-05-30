# Counterfactual Engine

## How it works

Relationship-aware counterfactual generation runs in three steps:

1. **Identify the intervention** — which protected attribute to flip and to what value (e.g. `sex: Male → Female`).
2. **Propagate through the DAG** — apply hard rules and sample learned mechanisms for every node that descends from the intervened attribute, in topological order.
3. **Validate** — check for hard-rule violations (`detect_constraint_violations`) and optionally run `ManifoldFilter` for soft distributional drift.

## The causal graph

```python
from relfair.graph import DependencyGraph

G = DependencyGraph.from_edges([
    ("sex", "relationship"),
    ("sex", "occupation"),
    ("age", "occupation"),
])
```

Nodes are attribute names. Edges are causal influence: `A → B` means B depends on A. The graph must be a DAG (cycles raise `ValueError`).

### Hard rules

Hard rules encode deterministic domain knowledge — things the data already know:

```python
# Unconditional: always set honorific=Ms when sex=Female
G.add_hard_rule("honorific", when={"sex": "Female"}, value="Ms")

# Conditional transition: only when current value IS Husband
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",      from_val="Husband")
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")
```

`from_val` means the rule fires **only** when the node's current value equals `from_val`. This is what prevents `Own-child` rows from being rewritten to `Wife`. Always add a passthrough rule for every value that should be preserved.

## Learned mechanisms

For nodes without hard rules, `LearnedMechanisms` fits one sklearn pipeline (ColumnTransformer → HistGBM) per non-root node:

```python
from relfair.mechanisms import LearnedMechanisms

mechs = LearnedMechanisms(G)
mechs.fit(training_df, cat_cols=["sex", "relationship", "occupation"])
```

`sample_batch` is the hot path — it makes one sklearn call per node for all rows simultaneously, which is ~700× faster than row-by-row sampling.

## Generating counterfactuals

```python
from relfair.counterfactual import batch_counterfactuals

# Naive: only flip the intervention attribute
naive = batch_counterfactuals(test_df, "sex", "Female", G, mechs, naive=True, seed=42)

# Relationship-aware: flip + propagate through descendants
relaware = batch_counterfactuals(test_df, "sex", "Female", G, mechs, naive=False, seed=42)
```

Both return a list of dicts (one per row). The naive path is kept for side-by-side comparison — it is the baseline every audit report should include.

## Detecting violations

```python
from relfair.counterfactual import detect_constraint_violations

violated = detect_constraint_violations(naive, G)
# Returns a boolean Series: True = this row violates a hard rule
```

This is a **deterministic, 100%-recall** detector for hard-rule incoherences. It checks every row against every hard rule in the graph. IsolationForest (`ManifoldFilter`) achieves ~7.5% recall for this class of violation — use it only for soft distributional drift, not hard-rule checks.

## Flip rate

```python
from relfair.counterfactual import flip_rate

rate = flip_rate(relaware, predict_fn)
```

`predict_fn` must be callable: `predict_fn(df) -> array-like of {0, 1}`. Returns the fraction of rows where the prediction changed between original and counterfactual.
