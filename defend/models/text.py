"""MiniLM pair-difference classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


@dataclass
class TextModel:
    seed: int = 20260827
    estimator: LogisticRegression | None = field(default=None, init=False)
    oof_: np.ndarray | None = field(default=None, init=False)

    def _new_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1_000,
            random_state=self.seed,
            solver="liblinear",
        )

    def fit(self, vectors: np.ndarray, present: np.ndarray, targets: Iterable[int]) -> "TextModel":
        y = np.asarray(list(targets), dtype=int)
        mask = np.asarray(present, dtype=bool)
        self.oof_ = np.zeros(len(y), dtype=float)
        selected = np.flatnonzero(mask)
        if len(selected) < 4 or len(np.unique(y[selected])) < 2:
            self.estimator = None
            return self
        selected_y = y[selected]
        folds = min(5, int(np.min(np.bincount(selected_y, minlength=2))))
        if folds >= 2:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
            for train_local, validation_local in splitter.split(vectors[selected], selected_y):
                estimator = self._new_estimator().fit(vectors[selected][train_local], selected_y[train_local])
                self.oof_[selected[validation_local]] = estimator.predict_proba(
                    vectors[selected][validation_local]
                )[:, 1]
        self.estimator = self._new_estimator().fit(vectors[selected], selected_y)
        return self

    def predict_proba(self, vectors: np.ndarray, present: np.ndarray) -> np.ndarray:
        output = np.zeros(len(vectors), dtype=float)
        mask = np.asarray(present, dtype=bool)
        if self.estimator is not None and np.any(mask):
            output[mask] = self.estimator.predict_proba(vectors[mask])[:, 1]
        return output
