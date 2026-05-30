# LL 144 Metrics

## What LL 144 requires

NYC Local Law 144 (effective July 2023) requires employers using automated employment decision tools (AEDTs) to conduct annual bias audits. The audit must report:

- **Selection rates** by sex and race/ethnicity
- **Impact ratios** (each group's rate ÷ the reference group's rate)
- **Four-fifths (80%) rule** flag when any ratio falls below 0.80
- **Intersectional** sex × race cross-tabs

relfair computes all of these with bootstrap confidence intervals.

## Usage

```python
from relfair.metrics import compute_ll144_metrics

result = compute_ll144_metrics(
    df,
    outcome_col="hired",          # column name for the binary or score outcome
    outcome_type="binary",        # "binary" (0/1) or "score" (continuous)
    sex_col="sex",                # column name for sex / gender
    race_col="race",              # column name for race / ethnicity
    n_boot=2000,                  # bootstrap resamples for CIs
    bootstrap_seed=42,
    reference_sex="Male",         # optional; highest-rate group used if omitted
    reference_race="White",       # optional; highest-rate group used if omitted
    meta={                        # passed through to the report
        "employer": "Acme Corp",
        "aedt": "Resume Ranker v4.2",
        "auditor": "Archit Rathod",
        "cover_period": "Jan 1 2025 – Dec 31 2025",
    },
)
```

## Result structure

```python
result.n_total          # int — total rows
result.n_selected       # int — rows with positive outcome
result.overall_rate     # float — overall selection rate

result.by_sex           # list[GroupStat]
result.by_race          # list[GroupStat]
result.intersectional   # list[IntersectionalStat]
```

### `GroupStat` fields

| Field | Type | Description |
|---|---|---|
| `group` | str | Group label (e.g. "Female", "Black") |
| `n` | int | Row count |
| `n_selected` | int | Selected count |
| `rate` | float | Selection rate |
| `ratio` | float | Rate ÷ reference rate |
| `ci_low`, `ci_high` | float | 95% bootstrap CI on the ratio |
| `p_value` | float | Two-proportion z-test vs. reference (None for reference group) |
| `four_fifths_flag` | bool | True if ratio < 0.80 |
| `small_sample_flag` | bool | True if n < 100 |

### `IntersectionalStat` fields

Same as `GroupStat` plus `sex` and `race` fields identifying the cell.

## Bootstrap CIs

CIs resample the **full dataset** (both reference and comparison group) so uncertainty in the reference group propagates correctly. This is more conservative than resampling only the comparison group — appropriate for regulatory reporting.

## Export

```python
from relfair.report import write_json, write_pdf, render_html

write_json(result, "report.json")   # machine-readable
write_pdf(result,  "report.pdf")    # DCWP-style PDF (requires WeasyPrint)
html = render_html(result)          # HTML string — no native deps
```
