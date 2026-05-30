"""
On-manifold validation via density / outlier filtering.

A counterfactual flip only counts as evidence of discrimination if the
counterfactual input is plausible — i.e. it lies on (or near) the data manifold.

This module provides belt-and-suspenders checks after the graph-propagation step.
IsolationForest is the default; a KDE-based scorer is an alternative for small datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder


class ManifoldFilter:
    """
    Fits an IsolationForest on training data and scores new inputs.

    Usage:
        filter = ManifoldFilter()
        filter.fit(training_df)
        mask = filter.is_on_manifold(counterfactual_df)
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 0) -> None:
        self._contamination = contamination
        self._random_state = random_state
        self._model: IsolationForest | None = None
        self._encoder: OrdinalEncoder | None = None
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []

    def fit(self, data: pd.DataFrame) -> None:
        self._cat_cols = [c for c in data.columns if data[c].dtype == object]
        self._num_cols = [c for c in data.columns if c not in self._cat_cols]

        X = self._encode(data)
        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=self._random_state,
        )
        self._model.fit(X)

    def is_on_manifold(self, data: pd.DataFrame) -> np.ndarray:
        """Returns a boolean array: True if the row is on-manifold."""
        if self._model is None:
            raise RuntimeError("Call fit() before is_on_manifold().")
        X = self._encode(data)
        # IsolationForest.predict returns 1 (inlier) or -1 (outlier)
        return self._model.predict(X) == 1

    def _encode(self, data: pd.DataFrame) -> np.ndarray:
        parts = []
        if self._num_cols:
            parts.append(data[self._num_cols].values.astype(float))
        if self._cat_cols:
            if self._encoder is None:
                self._encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                cat_enc = self._encoder.fit_transform(data[self._cat_cols])
            else:
                cat_enc = self._encoder.transform(data[self._cat_cols])
            parts.append(cat_enc)
        return np.hstack(parts) if parts else np.empty((len(data), 0))
