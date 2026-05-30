# `relfair.metrics`

## `compute_ll144_metrics`

```python
from relfair.metrics import compute_ll144_metrics

result = compute_ll144_metrics(
    df,
    outcome_col="hired",
    outcome_type="binary",     # "binary" (0/1) or "score" (continuous)
    sex_col="sex",
    race_col="race",
    n_boot=2000,
    bootstrap_seed=42,
    reference_sex=None,        # auto: highest-rate group
    reference_race=None,       # auto: highest-rate group
    meta={},                   # passed through to report
)
# Returns LL144Result
```

## `LL144Result`

| Field | Type |
|---|---|
| `n_total` | `int` |
| `n_selected` | `int` |
| `overall_rate` | `float` |
| `outcome_type` | `str` |
| `threshold` | `float \| None` |
| `by_sex` | `list[GroupStat]` |
| `by_race` | `list[GroupStat]` |
| `intersectional` | `list[IntersectionalStat]` |
| `reference_sex` | `str` |
| `reference_race` | `str` |
| `meta` | `dict` |

## `GroupStat`

| Field | Type |
|---|---|
| `group` | `str` |
| `n` | `int` |
| `n_selected` | `int` |
| `rate` | `float` |
| `ratio` | `float` |
| `ci_low` | `float` |
| `ci_high` | `float` |
| `p_value` | `float \| None` |
| `four_fifths_flag` | `bool` |
| `small_sample_flag` | `bool` |

## `IntersectionalStat`

All `GroupStat` fields plus:

| Field | Type |
|---|---|
| `sex` | `str` |
| `race` | `str` |

## Statistics helpers

```python
from relfair.metrics import bootstrap_ci, fisher_exact_pvalue

ci = bootstrap_ci(rates_array, n_boot=2000, seed=42)
# Returns (low, high) — 95% CI on the mean

p = fisher_exact_pvalue(n_a, k_a, n_b, k_b)
# Two-tailed Fisher exact test p-value
```
