# Contributing to relfair

Thanks for taking a look. `relfair` is the open-source counterfactual-fairness core of FairLens and a research artifact for an upcoming FAccT/AIES submission. Contributions that strengthen either side are welcome.

## Dev setup

```bash
git clone https://github.com/Archit1706/relfair
cd relfair
python -m venv .venv && source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Running checks

```bash
python -m pytest tests/        # ~32s, 62 tests
ruff check relfair/ tests/
mypy relfair/
```

CI runs the same three commands on Python 3.10, 3.11, and 3.12. A PR is mergeable when all three are green.

## What we welcome

- **Bug reports** with a minimal repro (a small DataFrame + the call that misbehaves).
- **New benchmark datasets** in `experiments/<name>/` following the existing pattern (`config.py`, `run.py`, `results/`). Datasets with known proxy variables (relationship → sex, occupation → sex/race, etc.) are highest-value.
- **Documentation fixes** — typos, clearer examples, missing edge cases.
- **Performance** improvements to `batch_counterfactuals` and `LearnedMechanisms.sample_batch` — both are hot paths.

## What needs design discussion first

Open an issue before sending a PR if you're touching:

- The `DependencyGraph` hard-rule semantics (`when` / `value` / `from_val`). Subtle to get right; see the `graph.py` docstrings for the failure modes.
- The public API of `compute_ll144_metrics` or `LL144Result`. Other tools depend on the shape.
- A new structure-discovery algorithm in `discovery.py`. We treat discovery output as *drafts for human review* — algorithms that imply ground truth aren't a fit.

## What's out of scope

- Real-time monitoring, multi-tenant infra, billing — those belong in a downstream product, not this library.
- BISG / probabilistic race imputation — different threat model.
- Individual-level (vs. group-level) counterfactuals via `dowhy.gcm` — possibly a future experiment, not core.

## Conventions

- **Reproducibility**: every generation/sampling function takes an explicit `seed`. No global RNG.
- **No framework deps in core**: don't import FastAPI, SQLAlchemy, Neo4j, etc. The whole point is that `relfair` is importable as a standalone research artifact.
- **Both code paths together**: keep naive and relationship-aware paths in `batch_counterfactuals(..., naive=True/False)` — side-by-side comparison is the headline exhibit.
- **Tests with new behaviour**: every new public function needs at least one test. The suite must stay under a minute.
- **Type hints** on all public functions. `mypy --strict` is enforced.

## Citation

If `relfair` informs research, please cite (paper in preparation):

```bibtex
@misc{rathod2026relfair,
  title  = {Relationship-Aware Counterfactual Fairness Testing},
  author = {Archit Rathod},
  year   = {2026},
  note   = {Preprint in preparation. Target venue: FAccT 2027 / AIES 2027.}
}
```

## Code of conduct

Be kind, be specific, assume good faith. Harassment in any form gets you removed from the project.
