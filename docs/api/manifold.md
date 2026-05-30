# `relfair.manifold`

## `ManifoldFilter`

```python
from relfair.manifold import ManifoldFilter
```

IsolationForest-based density filter. Use for **soft distributional violations** only — not for hard-rule incoherences (use `detect_constraint_violations` for those).

### Usage

```python
f = ManifoldFilter(contamination=0.05, random_state=0)
f.fit(training_df)

mask = f.is_on_manifold(counterfactual_df)
# np.ndarray[bool] — True = on-manifold
```

### Constructor

| Param | Default | Description |
|---|---|---|
| `contamination` | `0.05` | Expected fraction of outliers in training data |
| `random_state` | `0` | IsolationForest seed |

### When to use it

Use `ManifoldFilter` when:

- You have learned mechanisms (continuous or complex categorical) and want a secondary check that sampled values are in-distribution.
- Your graph has nodes with no hard rules and you want to flag unusual combinations.

Do **not** rely on it for hard-rule violations. Its recall for attribute-pair contradictions (e.g. `sex=Female, relationship=Husband`) is ~7.5% — effectively random.

### Encoding

Categorical columns (non-numeric dtype) are ordinally encoded. Numeric columns pass through. The encoder fitted during `fit()` is reused during `is_on_manifold()` so unknown categories are mapped to `-1`.
