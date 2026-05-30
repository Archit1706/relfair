"""
Causal structure discovery — proposes a draft dependency graph from observational data.

Uses causal-learn (PC algorithm by default; GES and NOTEARS available).

IMPORTANT: Causal discovery from observational data is genuinely unreliable.
The output of this module is a *draft* for human expert review, never ground truth.
The reviewed, version-pinned graph is what audits run against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def propose_graph(
    data: pd.DataFrame,
    method: str = "pc",
    alpha: float = 0.05,
) -> list[tuple[str, str]]:
    """
    Run structure discovery on *data* and return proposed directed edges.

    Args:
        data:   DataFrame of attribute values (protected + non-protected).
        method: 'pc' | 'ges' | 'notears'
        alpha:  Significance threshold for the PC algorithm.

    Returns:
        List of (src, dst) tuples representing proposed causal edges.
        This is a draft — it must be reviewed by a human before use in audits.
    """
    cols = list(data.columns)
    X = _encode(data)

    if method == "pc":
        edges = _run_pc(X, alpha=alpha)
    elif method == "ges":
        edges = _run_ges(X)
    elif method == "notears":
        edges = _run_notears(X)
    else:
        raise ValueError(f"Unknown discovery method: {method!r}. Choose 'pc', 'ges', or 'notears'.")

    return [(cols[i], cols[j]) for i, j in edges]


def _encode(data: pd.DataFrame) -> np.ndarray:
    cat_cols = [c for c in data.columns if data[c].dtype == object]
    num_cols = [c for c in data.columns if c not in cat_cols]
    parts = []
    if num_cols:
        parts.append(data[num_cols].values.astype(float))
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(data[cat_cols]))
    return np.hstack(parts) if parts else np.empty((len(data), 0))


def _run_pc(X: np.ndarray, alpha: float) -> list[tuple[int, int]]:
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except ImportError:
        raise ImportError("Install causal-learn: pip install causal-learn")

    cg = pc(X, alpha=alpha, indep_test=fisherz)
    edges = []
    n = X.shape[1]
    for i in range(n):
        for j in range(n):
            if cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1:
                edges.append((i, j))
    return edges


def _run_ges(X: np.ndarray) -> list[tuple[int, int]]:
    try:
        from causallearn.search.ScoreBased.GES import ges
    except ImportError:
        raise ImportError("Install causal-learn: pip install causal-learn")

    record = ges(X)
    edges = []
    n = X.shape[1]
    G = record["G"]
    for i in range(n):
        for j in range(n):
            if G.graph[i, j] == -1 and G.graph[j, i] == 1:
                edges.append((i, j))
    return edges


def _run_notears(X: np.ndarray) -> list[tuple[int, int]]:
    try:
        from causalnex.structure.notears import from_numpy
    except ImportError:
        raise ImportError("Install causalnex: pip install causalnex")

    sm = from_numpy(X, tabu_edges=[], w_threshold=0.3)
    return list(sm.edges)
