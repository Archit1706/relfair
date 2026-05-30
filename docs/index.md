# relfair

**Relationship-aware counterfactual fairness testing.**

[![CI](https://github.com/Archit1706/relfair/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Archit1706/relfair/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/relfair.svg)](https://pypi.org/project/relfair/)
[![Python](https://img.shields.io/pypi/pyversions/relfair.svg)](https://pypi.org/project/relfair/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Archit1706/relfair/blob/master/LICENSE)

---

## The problem with naive flip testing

Naive fairness tests flip one protected attribute — `sex: Male → Female` — and ask if the prediction changes. They miss the proxies.

A model that learned `relationship = Husband ⟹ Male` reads the flipped row as off-distribution and silently absorbs the bias. The flip never actually reaches the model's decision boundary — it just creates an incoherent input.

**relfair propagates the intervention through the causal graph.** When `sex` flips, every causally-dependent attribute flips with it: `Husband → Wife`, occupation distribution, household role. Counterfactuals stay on the data manifold. Discrimination that was invisible becomes measurable.

---

## Benchmark results

| Dataset | Naive flip rate | Rel-aware flip rate | Detection lift |
|---|---:|---:|---:|
| Adult — Husband rows (n=3,682) | 7.0% | 24.3% | **+17.2 pp** |
| ACS Income CA (n=5,241) | 7.5% | 34.5% | **+27.0 pp (4.6×)** |
| German Credit (n=144) | 4.9% | 13.2% | **+8.3 pp** |

Reproduce any row: `cd experiments/<dataset> && python run.py`

---

## What's included

**Counterfactual engine** — relationship-aware generation via a user-supplied causal DAG. Hard rules (e.g. `Husband → Wife`) plus learned structural equations (HistGBM per non-root node). Both naive and relationship-aware paths always present for side-by-side comparison.

**LL 144 metrics** — selection rates, impact ratios, the four-fifths rule, and intersectional sex × race cross-tabs with bootstrap CIs. DCWP-compliant PDF + JSON report output.

**CLI** — `relfair audit predictions.csv --outcome hired --sex sex --race race --pdf report.pdf`

---

## Install

```bash
pip install relfair                    # core
pip install "relfair[cli,report]"      # + CLI + PDF reports
pip install "relfair[all]"             # everything
```

[Get started →](quickstart.md)

---

## Paper

Preprint in preparation. Target venue: FAccT 2027 / AIES 2027.

```bibtex
@misc{rathod2026relfair,
  title  = {Relationship-Aware Counterfactual Fairness Testing},
  author = {Archit Rathod},
  year   = {2026},
  note   = {Preprint in preparation.}
}
```
