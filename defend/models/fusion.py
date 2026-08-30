"""Five-layer interpretable logistic fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class FusionModel:
    seed: int = 20260827
    estimator: LogisticRegression | None = field(default=None, init=False)

    def fit(self, layer_scores: np.ndarray, targets: Iterable[int]) -> "FusionModel":
        values = np.asarray(layer_scores, dtype=float)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("fusion requires exactly five layer scores")
        y = np.asarray(list(targets), dtype=int)
        if len(np.unique(y)) != 2:
            raise ValueError("fusion requires both classes")
        self.estimator = LogisticRegression(
            C=0.7,
            class_weight="balanced",
            max_iter=1_000,
            random_state=self.seed,
            solver="liblinear",
        ).fit(values, y)
        return self

    def predict_proba(self, layer_scores: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("fusion is not fitted")
        values = np.asarray(layer_scores, dtype=float)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("fusion requires exactly five layer scores")
        return self.estimator.predict_proba(values)[:, 1]
