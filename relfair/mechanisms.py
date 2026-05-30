"""
Structural equations for non-root nodes in the causal DAG.

Two mechanism types:
  Hard rules   — deterministic lookup tables defined in DependencyGraph.
  Learned      — one sklearn Pipeline per non-root node, fitted on training data.

LearnedMechanisms uses a ColumnTransformer to handle mixed numeric/categorical
parents cleanly and HistGradientBoosting models which need minimal tuning.

The optional *cat_cols* argument to fit() lets callers declare columns that are
categorical codes stored as floats (e.g. ACS OCCP=529 occupation codes, MAR=5
marital status codes).  Without it, only columns with object/category dtype are
treated as categorical — fine for the toy tests, wrong for ACS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


@dataclass
class _FittedMechanism:
    pipeline: Pipeline           # ColumnTransformer → HistGBM
    parent_cols: list[str]
    target_is_cat: bool
    target_enc: LabelEncoder | None    # None for continuous targets
    residual_std: float                # noise for continuous sampling


def _is_cat(col: str, data: pd.DataFrame, explicit: set[str]) -> bool:
    return col in explicit or data[col].dtype == object or str(data[col].dtype) == "category"


def _build_pipeline(
    data: pd.DataFrame,
    parent_cols: list[str],
    target_is_cat: bool,
    explicit_cat: set[str],
) -> Pipeline:
    cat_cols = [c for c in parent_cols if _is_cat(c, data, explicit_cat)]
    num_cols = [c for c in parent_cols if c not in cat_cols]

    transformers: list[tuple] = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_cols,
        ))

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    # early_stopping=False prevents the internal stratified val-split that
    # fails when some target classes have < 2 samples (e.g. rare OCCP codes).
    estimator: Any = (
        HistGradientBoostingClassifier(max_iter=100, random_state=0, early_stopping=False)
        if target_is_cat
        else HistGradientBoostingRegressor(max_iter=100, random_state=0, early_stopping=False)
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def _to_python(val: Any) -> Any:
    """Coerce numpy/pandas scalars to plain Python types for clean dict storage."""
    if hasattr(val, "item"):      # numpy scalar
        return val.item()
    if hasattr(val, "tolist"):    # 0-d array or pandas Categorical scalar
        return val.tolist()
    return val


class LearnedMechanisms:
    """
    Fits one structural equation (sklearn Pipeline) per non-root node.

    Args (to fit):
        data:     Training DataFrame. Must include all columns referenced by
                  the DependencyGraph.
        cat_cols: Explicit list of column names to treat as categorical even if
                  their dtype is float64 (e.g. integer-coded ACS variables like
                  OCCP, MAR, SEX, RAC1P).  Columns with object or category dtype
                  are always treated as categorical regardless.

    Usage:
        mechs = LearnedMechanisms(G)
        mechs.fit(training_df, cat_cols=["SEX", "MAR", "OCCP", ...])
        val = mechs.sample("MAR", {"SEX": 2.0, "AGEP": 34.0}, rng)
    """

    def __init__(self, graph: Any) -> None:   # graph: DependencyGraph
        self._graph = graph
        self._mechs: dict[str, _FittedMechanism] = {}
        self._explicit_cat: set[str] = set()

    def fit(self, data: pd.DataFrame, cat_cols: list[str] | None = None) -> None:
        """Fit one mechanism per non-root node."""
        self._explicit_cat = set(cat_cols or [])

        for node in self._graph.nodes():
            parents = self._graph.parents(node)
            if not parents:
                continue

            y = data[node]
            target_is_cat = _is_cat(node, data, self._explicit_cat)

            target_enc: LabelEncoder | None = None
            if target_is_cat:
                target_enc = LabelEncoder()
                y_enc = target_enc.fit_transform(y.astype(str))
            else:
                y_enc = y.values.astype(float)

            pipe = _build_pipeline(data, parents, target_is_cat, self._explicit_cat)
            pipe.fit(data[parents], y_enc)

            residual_std = 0.0
            if not target_is_cat:
                preds = pipe.predict(data[parents])
                residual_std = float(np.std(y_enc - preds))

            self._mechs[node] = _FittedMechanism(
                pipeline=pipe,
                parent_cols=list(parents),
                target_is_cat=target_is_cat,
                target_enc=target_enc,
                residual_std=residual_std,
            )

    def sample(
        self, node: str, parent_values: dict[str, Any], rng: np.random.Generator
    ) -> Any:
        """Sample a value for *node* given *parent_values*."""
        mech = self._mechs.get(node)
        if mech is None:
            raise LookupError(f"No fitted mechanism for node '{node}'.")

        X = pd.DataFrame([{c: parent_values[c] for c in mech.parent_cols}])

        if mech.target_is_cat:
            proba = mech.pipeline.predict_proba(X)[0]
            idx = int(rng.choice(len(proba), p=proba))
            assert mech.target_enc is not None
            raw = mech.target_enc.inverse_transform([idx])[0]
            # inverse_transform returns the string we encoded — convert back to
            # original numeric type when the column was a float-coded categorical
            if node in self._explicit_cat:
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    return _to_python(raw)
            return _to_python(raw)
        else:
            pred = float(mech.pipeline.predict(X)[0])
            noise = float(rng.normal(0.0, mech.residual_std)) if mech.residual_std > 0 else 0.0
            return pred + noise

    def sample_batch(
        self, node: str, parent_df: pd.DataFrame, rng: np.random.Generator
    ) -> list[Any]:
        """
        Vectorized batch sample — one prediction call for all rows in *parent_df*.

        Much faster than calling sample() row-by-row when processing large test sets.
        Returns a list of sampled values, one per row, in the same order as *parent_df*.
        """
        mech = self._mechs.get(node)
        if mech is None:
            raise LookupError(f"No fitted mechanism for node '{node}'.")

        X = parent_df[mech.parent_cols]
        n = len(X)

        if mech.target_is_cat:
            proba = mech.pipeline.predict_proba(X)          # (n, n_classes)
            # Vectorised multinomial sampling — equivalent to rng.choice per row
            cumsum = proba.cumsum(axis=1)
            r = rng.random(n)[:, np.newaxis]
            indices = (r >= cumsum).sum(axis=1).clip(0, proba.shape[1] - 1)
            assert mech.target_enc is not None
            raw_vals = mech.target_enc.inverse_transform(indices)
            if node in self._explicit_cat:
                # ACS uses float-coded categoricals (OCCP=3.0, MAR=1.0 …).
                # Adult uses string categoricals ("Unmarried", …).  Try float
                # conversion but fall back to native type when it fails.
                out = []
                for v in raw_vals:
                    try:
                        out.append(float(v))
                    except (ValueError, TypeError):
                        out.append(_to_python(v))
                return out
            return [_to_python(v) for v in raw_vals]
        else:
            preds = mech.pipeline.predict(X).astype(float)
            if mech.residual_std > 0:
                noise = rng.normal(0.0, mech.residual_std, n)
                preds = preds + noise
            return [float(v) for v in preds]

    def fitted_nodes(self) -> list[str]:
        return list(self._mechs.keys())
