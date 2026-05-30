# `relfair.discovery`

## `propose_graph`

```python
from relfair.discovery import propose_graph

G_draft = propose_graph(
    df,
    method="pc",          # "pc" | "ges" | "notears"
    protected=["sex", "race"],
)
```

Runs causal structure discovery and returns a `DependencyGraph`.

!!! warning "Draft only — always review before use"
    Structure discovery algorithms return a graph that is **statistically consistent with the data**.
    They do not know domain semantics. A discovered edge `age → sex` is statistically possible
    and semantically wrong. Always have a domain expert review and edit the output before
    using it in an audit.

### Methods

| `method` | Algorithm | Notes |
|---|---|---|
| `"pc"` | PC algorithm | Fast; assumes faithfulness |
| `"ges"` | GES (Greedy Equivalence Search) | Better on smaller datasets |
| `"notears"` | NOTEARS (continuous optimisation) | Dense graphs; slower |

### Workflow

```python
# 1. Get a draft
G_draft = propose_graph(df, method="pc", protected=["sex", "race"])

# 2. Inspect — print edges for review
for parent, child in G_draft.edges():
    print(f"  {parent} → {child}")

# 3. Edit manually — add missing domain edges, remove spurious ones
G_draft.add_edge("sex", "relationship")
G_draft.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife", from_val="Husband")

# 4. Only then use for generation
mechs = LearnedMechanisms(G_draft)
mechs.fit(training_df)
```
