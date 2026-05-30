# Changelog

## v0.1.0 — 2026-05-29

Initial public release.

**Counterfactual engine**

- `DependencyGraph` — DAG with hard rules and conditional (`from_val`) transitions
- `LearnedMechanisms` — HistGBM structural equations per non-root node, vectorised batch sampling
- `batch_counterfactuals` — naive and relationship-aware generation
- `detect_constraint_violations` — deterministic, 100%-recall hard-rule checker
- `ManifoldFilter` — IsolationForest for soft distributional drift

**LL 144 metrics**

- `compute_ll144_metrics` — selection rates, impact ratios, four-fifths rule, intersectional sex × race, bootstrap CIs
- PDF (WeasyPrint + Jinja2), HTML, and JSON report export
- `relfair audit` CLI

**Benchmarks**

- Adult/Census: +17.2 pp detection lift
- ACS Income CA: +27.0 pp (4.6×) detection lift
- German Credit: +8.3 pp detection lift

**Structure discovery**

- `propose_graph` — PC / GES / NOTEARS draft graph generation (human review required)
