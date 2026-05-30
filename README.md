# relfair

**Relationship-aware counterfactual fairness testing.**

[![CI](https://github.com/Archit1706/relfair/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Archit1706/relfair/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/relfair.svg)](https://pypi.org/project/relfair/)
[![Python](https://img.shields.io/pypi/pyversions/relfair.svg)](https://pypi.org/project/relfair/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Naive fairness tests flip one protected attribute (e.g. `sex: Male → Female`) and ask if the prediction changes. They miss the proxies: a model that learns `relationship = Husband ⇒ Male` will read the flipped row as off-distribution and silently absorb the bias. `relfair` propagates the intervention through the causal graph — `Husband → Wife`, occupation, household role — so counterfactuals stay on the data manifold. **Result: 3–4× more discrimination detected on Adult, ACS, and German Credit.**

The library also ships an NYC Local Law 144 audit engine: selection rates, impact ratios, the four-fifths rule, intersectional sex × race cross-tabs with bootstrap CIs, and DCWP-compliant PDF/JSON reports.

> Status: **alpha** (`0.1.0`). The counterfactual engine and LL 144 metrics are stable and covered by 62 tests. The public API may still change before `1.0`.

---

## Why this exists

| Dataset | Naive flip rate | Rel-aware flip rate | Detection lift |
|---|---:|---:|---:|
| Adult — Husband rows (n=3,682) | 7.0% | 24.3% | **+17.2 pp** |
| ACS Income CA (n=5,241)        | 7.5% | 34.5% | **+27.0 pp (4.6×)** |
| German Credit (n=144)          | 4.9% | 13.2% | **+8.3 pp** |

Reproduce: `cd experiments/<dataset> && python run.py`.

A separate finding: `detect_constraint_violations()` has **100% recall** for hard-rule incoherences (a row with `sex=Female, relationship=Husband` never occurs in training data and must be off-manifold). IsolationForest flags ~7.5% of rows uniformly regardless of whether the violation is present — it cannot localise attribute-pair contradictions. Use the constraint check for hard rules; reserve the IsolationForest filter for soft distributional drift.

---

## Install

```bash
pip install relfair                       # core
pip install "relfair[cli,report]"         # + CLI + PDF reports
pip install "relfair[experiments]"        # + benchmark deps
pip install "relfair[all]"                # everything
```

PDF generation uses WeasyPrint, which needs the GTK3 runtime on Windows. The CLI's `--html` flag produces a report without native deps. See [WeasyPrint installation](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html).

---

## Quick start — counterfactual engine

```python
from relfair.graph import DependencyGraph
from relfair.mechanisms import LearnedMechanisms
from relfair.counterfactual import batch_counterfactuals, detect_constraint_violations

G = DependencyGraph.from_edges([
    ("sex", "relationship"),
    ("sex", "occupation"),
    ("age", "occupation"),
])

# Conditional transition: flip Husband → Wife only when current value is Husband.
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",      from_val="Husband")
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")

mechs = LearnedMechanisms(G)
mechs.fit(training_df, cat_cols=["sex", "relationship", "occupation"])

naive    = batch_counterfactuals(test_df, "sex", "Female", G, mechs, naive=True,  seed=42)
relaware = batch_counterfactuals(test_df, "sex", "Female", G, mechs, naive=False, seed=42)

violated = detect_constraint_violations(naive, G)   # definitionally off-manifold
```

> **`from_val` is load-bearing.** Without it, a rule "when sex=Female → relationship=Wife" overwrites every relationship to `Wife`. Always add passthrough rules (e.g. `Own-child → Own-child`) for values that should be preserved.

---

## Quick start — LL 144 audit CLI

```bash
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

Produces: a console table (rates, ratios, four-fifths flags, bootstrap CIs), a DCWP-style PDF, and machine-readable JSON with all metrics, CIs, p-values, and flags.

---

## Quick start — metrics API

```python
from relfair.metrics import compute_ll144_metrics
from relfair.report   import write_json, write_pdf

result = compute_ll144_metrics(
    df,
    outcome_col="hired",
    outcome_type="binary",            # or "score"
    sex_col="sex",
    race_col="race",
    n_boot=2000,
    bootstrap_seed=42,
    meta={"employer": "Acme Corp", "auditor": "Archit Rathod"},
)

for group in result.by_race:
    print(f"{group.group}: ratio={group.ratio:.3f}  flag={group.four_fifths_flag}")

write_json(result, "report.json")
write_pdf(result,  "report.pdf")      # WeasyPrint + GTK3
```

---

## Module map

```
relfair/
  graph.py           DependencyGraph — DAG, topological ops, hard rules (incl. from_val)
  mechanisms.py      LearnedMechanisms — ColumnTransformer + HistGBM per non-root node
  counterfactual.py  batch_counterfactuals, detect_constraint_violations, flip_rate
  manifold.py        ManifoldFilter — IsolationForest for soft distributional violations
  discovery.py       propose_graph — PC/GES/NOTEARS structure discovery (draft only)
  metrics/
    ll144.py         compute_ll144_metrics → LL144Result (GroupStat, IntersectionalStat)
    stats.py         bootstrap_impact_ratio_cis, fisher_exact_pvalue, two_proportion_ztest
  report/
    pdf.py           render_html, render_pdf, write_pdf
    json_export.py   to_dict, to_json, write_json
    templates/
      ll144.html.j2  DCWP-style Jinja2 template
  cli.py             Click CLI — `relfair audit ...`

experiments/
  adult/             Adult/Census — Husband/Wife constraint, +17.2 pp
  folktables/        ACS Income CA — occupation proxy, +27.0 pp (4.6×)
  german_credit/     personal_status hard rule, +8.3 pp

tests/               62 tests (pytest, ~32 s)
```

---

## Limitations and assumptions

- **Group-level, not individual-level.** `relfair` measures aggregate disparities under intervention. For per-row counterfactual claims you want `dowhy.gcm` or similar — a future experiment, not core.
- **Causal graph is an input, not a discovery.** `discovery.propose_graph()` exists, but its output is a *draft for human review*. Hard rules and edges should always be validated by a domain expert before audit results are trusted.
- **Hard rules require care.** Conditional transitions (`from_val`) only fire when the current value matches. Forgetting to add passthrough rules for values that should be preserved silently rewrites them. See [`graph.py`](relfair/graph.py) docstrings.
- **Race categories are categorical labels you pass in.** No BISG / probabilistic imputation; that's a different threat model.
- **Two protected attributes in the LL 144 path.** Sex and race only, per the regulation. Generalising to additional axes is straightforward but not yet implemented.

---

## Running tests

```bash
pip install -e ".[dev]"

python -m pytest tests/                          # full suite, ~32 s
python -m pytest tests/test_metrics.py -v        # single file
python -m pytest tests/ -k "test_from_val_rule"  # single test by name
```

CI runs the same on Python 3.10, 3.11, and 3.12.

---

## Design constraints

- **No framework dependencies in the core.** No FastAPI, no SQLAlchemy, no Neo4j driver — `relfair` is importable as a standalone research artifact.
- **Reproducibility.** Every generation and sampling function accepts an explicit `seed`. No global RNG mutation.
- **Both code paths always present.** `batch_counterfactuals(..., naive=True/False)` — the headline exhibit is naive vs. relationship-aware side-by-side.
- **`cat_cols` for float-coded categoricals.** ACS encodes categories as floats (OCCP, MAR, SEX). Pass `cat_cols=[...]` to `LearnedMechanisms.fit()` so they're treated correctly.

---

## Citation

Paper in preparation. Target venue: FAccT 2027 / AIES 2027.

```bibtex
@misc{rathod2026relfair,
  title  = {Relationship-Aware Counterfactual Fairness Testing},
  author = {Archit Rathod},
  year   = {2026},
  note   = {Preprint in preparation.}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports with minimal repros, new benchmark datasets, and documentation fixes are especially welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE).
