# `relfair.counterfactual`

## `batch_counterfactuals`

```python
from relfair.counterfactual import batch_counterfactuals

records = batch_counterfactuals(
    df, "sex", "Female", G, mechs,
    naive=False,   # True = flip only; False = flip + propagate
    seed=42,
)
```

Returns a `list[dict]` — one dict per row with all column values after intervention.

- `naive=True` — only flips the intervention attribute; all descendants unchanged.
- `naive=False` — flips + propagates through every descendant via hard rules and learned mechanisms.

Always generate both and compare — the side-by-side is the headline exhibit of every audit.

## `detect_constraint_violations`

```python
from relfair.counterfactual import detect_constraint_violations

violated = detect_constraint_violations(records, G)
# pd.Series[bool] — True = row violates a hard rule
```

Checks every record against every hard rule in the graph. Deterministic, 100% recall for hard-rule incoherences. Superior to `ManifoldFilter` for this class of violation.

## `flip_rate`

```python
from relfair.counterfactual import flip_rate

rate = flip_rate(records, predict_fn)
# float — fraction of rows where prediction changed
```

`predict_fn(df) -> array-like of {0, 1}`. The function is called once on a DataFrame built from `records`.

## `generate_counterfactual`

```python
from relfair.counterfactual import generate_counterfactual

cf = generate_counterfactual(
    row,          # dict — a single original row
    "sex", "Female",
    G, mechs,
    naive=False,
    rng=np.random.default_rng(42),
)
```

Single-row version. Use `batch_counterfactuals` for anything beyond debugging — it is vectorised and much faster.
