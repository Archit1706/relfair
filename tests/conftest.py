"""
Shared test fixtures.

The toy dataset mimics the Adult/Census structure:
  sex        ∈ {M, F}           — protected attribute
  honorific  ∈ {Mr, Ms}         — hard rule: sex=M→Mr, sex=F→Ms
  marital    ∈ {Single, Married} — soft dependency on sex
  income_k   — continuous, soft dependency on sex
  hired      — binary label (target; not a graph node)

DAG:  sex ──→ honorific
      sex ──→ marital
      sex ──→ income_k

Hard rules on honorific:
  sex=M  →  Mr
  sex=F  →  Ms
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relfair.graph import DependencyGraph


# ---------------------------------------------------------------------------
# Graph fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph() -> DependencyGraph:
    G = DependencyGraph.from_edges([
        ("sex", "honorific"),
        ("sex", "marital"),
        ("sex", "income_k"),
    ])
    G.add_hard_rule("honorific", when={"sex": "M"}, value="Mr")
    G.add_hard_rule("honorific", when={"sex": "F"}, value="Ms")
    return G


# ---------------------------------------------------------------------------
# Dataset fixture  (300 rows, reproducible)
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300

    sex = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])

    # honorific is deterministic
    honorific = np.where(sex == "M", "Mr", "Ms")

    # marital: mostly Married for M, Single for F (with noise)
    marital = np.where(
        sex == "M",
        rng.choice(["Married", "Single"], size=n, p=[0.70, 0.30]),
        rng.choice(["Married", "Single"], size=n, p=[0.40, 0.60]),
    )

    # income_k: M ~ N(55, 10), F ~ N(48, 10)
    income_k = np.where(
        sex == "M",
        rng.normal(55, 10, n),
        rng.normal(48, 10, n),
    ).clip(20, 100)

    # binary label correlated with income
    p_hired = 1 / (1 + np.exp(-(income_k - 50) / 10))
    hired = (rng.random(n) < p_hired).astype(int)

    return pd.DataFrame({
        "sex": sex,
        "honorific": honorific,
        "marital": marital,
        "income_k": income_k.round(2),
        "hired": hired,
    })
