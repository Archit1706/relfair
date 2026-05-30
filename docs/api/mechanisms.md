# `relfair.mechanisms`

## `LearnedMechanisms`

```python
from relfair.mechanisms import LearnedMechanisms
```

Fits one sklearn Pipeline (ColumnTransformer → HistGBM) per non-root node in the graph.

### Methods

#### `fit(data, cat_cols=None)`

```python
mechs = LearnedMechanisms(G)
mechs.fit(training_df, cat_cols=["sex", "relationship", "occupation"])
```

- `data` — `pd.DataFrame`. Must contain all columns referenced in the graph.
- `cat_cols` — columns to treat as categorical even if their dtype is numeric (e.g. ACS float-coded variables like `OCCP=3.0`, `MAR=5.0`). Columns with `object` or `category` dtype are always treated as categorical.

#### `sample(node, parent_values, rng)`

```python
val = mechs.sample("occupation", {"sex": "Female", "age": 34.0}, rng)
```

Sample one value for `node` given `parent_values`. `rng` must be a `np.random.Generator`.

#### `sample_batch(node, parent_df, rng)`

```python
vals = mechs.sample_batch("occupation", parent_df, rng)
```

Vectorised batch sample — one sklearn call for all rows. ~700× faster than row-by-row `sample()`. Returns a `list` of values in the same order as `parent_df`.

#### `fitted_nodes()`

Returns the list of nodes for which a mechanism was fitted (all non-root nodes).

### Internals

- Categorical parent columns: `OrdinalEncoder(handle_unknown="use_encoded_value")`
- Numeric parent columns: passed through unchanged
- Categorical targets: `HistGradientBoostingClassifier` + `LabelEncoder`
- Continuous targets: `HistGradientBoostingRegressor` + Gaussian noise scaled to residual std
- `early_stopping=False` to avoid stratified val-split failures on rare target classes
