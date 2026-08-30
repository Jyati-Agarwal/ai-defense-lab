"""Deterministic CPU LightGBM with five-fold out-of-fold isotonic calibration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class CalibratedLightGBM:
    feature_names: tuple[str, ...]
    seed: int = 20260827
    estimator: LGBMClassifier | None = field(default=None, init=False)
    calibrator: IsotonicRegression | None = field(default=None, init=False)
    oof_calibrated_: np.ndarray | None = field(default=None, init=False)
    oof_calibration_fold_: np.ndarray | None = field(default=None, init=False, repr=False)
    oof_calibration_roles_: np.ndarray | None = field(default=None, init=False, repr=False)
    oof_calibration_audit_: tuple[dict[str, Any], ...] = field(default_factory=tuple, init=False, repr=False)
    final_calibration_roles_: np.ndarray | None = field(default=None, init=False, repr=False)
    final_calibration_audit_: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    cv_pr_auc_: float | None = field(default=None, init=False)
    cv_folds_: int = field(default=0, init=False)
    _explainer: Any | None = field(default=None, init=False, repr=False)

    def _new_estimator(self, y: np.ndarray) -> LGBMClassifier:
        positives = max(1, int(np.sum(y == 1)))
        negatives = max(1, int(np.sum(y == 0)))
        return LGBMClassifier(
            objective="binary",
            n_estimators=90,
            learning_rate=0.055,
            num_leaves=15,
            max_depth=5,
            min_child_samples=max(5, min(30, len(y) // 40)),
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=negatives / positives,
            random_state=self.seed,
            bagging_seed=self.seed,
            feature_fraction_seed=self.seed,
            data_random_seed=self.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )

    def fit(self, frame: pd.DataFrame | np.ndarray, targets: Iterable[int]) -> "CalibratedLightGBM":
        self._explainer = None
        values = self._values(frame)
        y = np.asarray(list(targets), dtype=int)
        counts = np.bincount(y, minlength=2)
        if len(y) != len(values) or np.min(counts) < 2:
            raise ValueError("calibrated LightGBM needs at least two examples of each class")
        folds = min(5, int(np.min(counts)))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        fold_assignments = np.full(len(y), -1, dtype=np.int8)
        roles = np.zeros((folds, len(y)), dtype=np.int8)
        calibrated = np.zeros(len(y), dtype=float)
        calibration_audit = []
        split_positions = list(splitter.split(values, y))
        for fold, (outer_training_positions, validation_positions) in enumerate(split_positions):
            calibration_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.20,
                random_state=self.seed + 10_000 + fold,
            )
            estimator_local, calibration_local = next(
                calibration_splitter.split(values[outer_training_positions], y[outer_training_positions])
            )
            estimator_positions = outer_training_positions[estimator_local]
            calibration_positions = outer_training_positions[calibration_local]
            roles[fold, estimator_positions] = 1
            roles[fold, calibration_positions] = 2
            roles[fold, validation_positions] = 3
            fold_assignments[validation_positions] = fold
            estimator = self._new_estimator(y[estimator_positions])
            estimator.fit(values[estimator_positions], y[estimator_positions], feature_name=list(self.feature_names))
            calibration_raw = estimator.predict_proba(values[calibration_positions])[:, 1]
            fold_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                calibration_raw,
                y[calibration_positions],
            )
            validation_raw = estimator.predict_proba(values[validation_positions])[:, 1]
            calibrated[validation_positions] = fold_calibrator.predict(validation_raw)
            pairwise_overlap_count = sum((
                len(np.intersect1d(estimator_positions, calibration_positions)),
                len(np.intersect1d(estimator_positions, validation_positions)),
                len(np.intersect1d(calibration_positions, validation_positions)),
            ))
            calibration_audit.append({
                "fold": fold,
                "estimator_fit_target_count": len(estimator_positions),
                "calibration_target_count": len(calibration_positions),
                "validation_target_count": len(validation_positions),
                "estimator_fit_positions_sha256": hashlib.sha256(
                    np.sort(estimator_positions).astype(np.int64, copy=False).tobytes()
                ).hexdigest(),
                "calibration_positions_sha256": hashlib.sha256(
                    np.sort(calibration_positions).astype(np.int64, copy=False).tobytes()
                ).hexdigest(),
                "validation_positions_sha256": hashlib.sha256(
                    np.sort(validation_positions).astype(np.int64, copy=False).tobytes()
                ).hexdigest(),
                "pairwise_overlap_count": pairwise_overlap_count,
                "coverage_count": len(np.unique(np.r_[estimator_positions, calibration_positions, validation_positions])),
            })
        self.oof_calibrated_ = calibrated
        self.oof_calibration_fold_ = fold_assignments
        self.oof_calibration_roles_ = roles
        self.oof_calibration_audit_ = tuple(calibration_audit)
        self.cv_pr_auc_ = float(average_precision_score(y, self.oof_calibrated_))
        self.cv_folds_ = folds
        all_positions = np.arange(len(y), dtype=int)
        final_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=self.seed + 20_000)
        final_estimator_local, final_calibration_local = next(final_splitter.split(values, y))
        final_estimator_positions = all_positions[final_estimator_local]
        final_calibration_positions = all_positions[final_calibration_local]
        final_roles = np.zeros(len(y), dtype=np.int8)
        final_roles[final_estimator_positions] = 1
        final_roles[final_calibration_positions] = 2
        self.estimator = self._new_estimator(y[final_estimator_positions])
        self.estimator.fit(
            values[final_estimator_positions],
            y[final_estimator_positions],
            feature_name=list(self.feature_names),
        )
        final_calibration_raw = self.estimator.predict_proba(values[final_calibration_positions])[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            final_calibration_raw,
            y[final_calibration_positions],
        )
        self.final_calibration_roles_ = final_roles
        final_overlap_count = len(np.intersect1d(final_estimator_positions, final_calibration_positions))
        self.final_calibration_audit_ = {
            "estimator_fit_target_count": len(final_estimator_positions),
            "calibration_target_count": len(final_calibration_positions),
            "estimator_fit_positions_sha256": hashlib.sha256(
                np.sort(final_estimator_positions).astype(np.int64, copy=False).tobytes()
            ).hexdigest(),
            "calibration_positions_sha256": hashlib.sha256(
                np.sort(final_calibration_positions).astype(np.int64, copy=False).tobytes()
            ).hexdigest(),
            "overlap_count": final_overlap_count,
            "coverage_count": len(y),
        }
        return self

    def _values(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame, pd.DataFrame):
            return frame.loc[:, self.feature_names].to_numpy(dtype=float)
        values = np.asarray(frame, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix does not match trained feature names")
        return values

    def predict_proba(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.estimator is None or self.calibrator is None:
            raise RuntimeError("model is not fitted")
        raw = self.estimator.predict_proba(self._values(frame))[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def shap_values(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        values = self._values(frame)
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.estimator.booster_)
        explained = self._explainer.shap_values(values)
        if isinstance(explained, list):
            explained = explained[-1]
        array = np.asarray(explained)
        if array.ndim == 3:
            array = array[:, :, -1]
        return array.astype(float, copy=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None
        return state
