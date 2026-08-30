import hashlib

import numpy as np
import pandas as pd

import defend.models.tabular as tabular_module
from defend.features import FEATURE_COLUMNS
from defend.features.pipeline import LAYER_COLUMNS, FeatureBatch
from defend.models import DefenseModel
from defend.models.bundle import REASON_MESSAGES


def test_lightgbm_isotonic_fusion_and_tree_shap_reason_codes(monkeypatch) -> None:
    rng = np.random.default_rng(20260827)
    count = 120
    targets = np.zeros(count, dtype=int)
    targets[::5] = 1
    frame = pd.DataFrame(rng.normal(0, 0.08, (count, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["injection_fingerprint_score"] = np.clip(targets * 0.9 + rng.normal(0, 0.05, count), 0, 1)
    frame["contextualized_semantic_drift"] = np.clip(targets * 0.8 + rng.normal(0, 0.05, count), 0, 1)
    frame["amount_zscore_entity"] = targets * 4 + rng.random(count)
    frame["price_vs_market_median"] = 1 + targets * 2 + rng.random(count) * 0.1
    frame["structural_floor"] = 0
    frame["credit_utilisation_regime_change"] = 0
    frame["layer_structural"] = 0
    frame["layer_semantic"] = frame["contextualized_semantic_drift"]
    frame["layer_provenance"] = frame["injection_fingerprint_score"]
    frame["layer_temporal"] = np.clip(frame["amount_zscore_entity"] / 5, 0, 1)
    frame["layer_economic"] = np.clip((frame["price_vs_market_median"] - 1) / 3, 0, 1)
    vectors = rng.normal(size=(count, 384)).astype("float32")
    vectors += targets[:, None] * 0.25
    batch = FeatureBatch(frame, vectors, np.ones(count, dtype=bool), [{} for _ in range(count)])

    model = DefenseModel().fit(batch, targets)
    original_explainer = tabular_module.shap.TreeExplainer
    explainer_constructions = []

    def counting_explainer(*args, **kwargs):
        explainer_constructions.append(1)
        return original_explainer(*args, **kwargs)

    monkeypatch.setattr(tabular_module.shap, "TreeExplainer", counting_explainer)
    result = model.predict(batch)
    first_construction_count = len(explainer_constructions)

    assert model.tree is not None and model.tree.cv_folds_ == 5
    assert model.tree.calibrator is not None
    assert model.fusion is not None and model.fusion.estimator.coef_.shape == (1, 5)
    assert np.asarray(result["layers"]).shape == (count, 5)
    assert all(len(reasons) == 3 for reasons in result["reason_codes"])
    assert np.mean(np.asarray(result["fused"])[targets == 1]) > np.mean(np.asarray(result["fused"])[targets == 0])
    assert 1 <= first_construction_count <= 4
    model.predict(batch)
    assert len(explainer_constructions) == first_construction_count

    for calibrated_model in [model.tree, *model.layer_models.values()]:
        assignments = calibrated_model.oof_calibration_fold_
        roles = calibrated_model.oof_calibration_roles_
        assert assignments is not None and np.all(assignments >= 0)
        assert roles is not None and roles.shape == (calibrated_model.cv_folds_, count)
        for audit in calibrated_model.oof_calibration_audit_:
            fold = int(audit["fold"])
            estimator_positions = np.flatnonzero(roles[fold] == 1)
            calibration_positions = np.flatnonzero(roles[fold] == 2)
            validation_positions = np.flatnonzero(roles[fold] == 3)
            assert np.array_equal(validation_positions, np.flatnonzero(assignments == fold))
            assert not np.intersect1d(estimator_positions, calibration_positions).size
            assert not np.intersect1d(estimator_positions, validation_positions).size
            assert not np.intersect1d(calibration_positions, validation_positions).size
            assert audit["pairwise_overlap_count"] == 0
            assert audit["coverage_count"] == count
            assert audit["estimator_fit_target_count"] == len(estimator_positions)
            assert audit["validation_target_count"] == len(validation_positions)
            assert audit["calibration_target_count"] == len(calibration_positions)
            assert audit["estimator_fit_positions_sha256"] == hashlib.sha256(
                estimator_positions.astype(np.int64).tobytes()
            ).hexdigest()
            assert audit["validation_positions_sha256"] == hashlib.sha256(
                validation_positions.astype(np.int64).tobytes()
            ).hexdigest()
            assert audit["calibration_positions_sha256"] == hashlib.sha256(
                calibration_positions.astype(np.int64).tobytes()
            ).hexdigest()
        final_roles = calibrated_model.final_calibration_roles_
        assert final_roles is not None
        final_estimator_positions = np.flatnonzero(final_roles == 1)
        final_calibration_positions = np.flatnonzero(final_roles == 2)
        final_audit = calibrated_model.final_calibration_audit_
        assert not np.intersect1d(final_estimator_positions, final_calibration_positions).size
        assert final_audit["overlap_count"] == 0
        assert final_audit["coverage_count"] == count
        assert final_audit["estimator_fit_positions_sha256"] == hashlib.sha256(
            final_estimator_positions.astype(np.int64).tobytes()
        ).hexdigest()
        assert final_audit["calibration_positions_sha256"] == hashlib.sha256(
            final_calibration_positions.astype(np.int64).tobytes()
        ).hexdigest()

    # Reasons must come from the active layer/fusion path, not the unused
    # all-feature diagnostic tree.
    model.tree = object()  # type: ignore[assignment]
    assert all(len(reasons) == 3 for reasons in model.predict(batch)["reason_codes"])

    structural_frame = frame.iloc[[0]].copy()
    structural_frame["structural_floor"] = 1.0
    structural_batch = FeatureBatch(
        structural_frame,
        vectors[:1],
        np.ones(1, dtype=bool),
        [{}],
    )
    structural = model.predict(structural_batch)
    assert structural["fused"][0] == 1.0
    assert structural["reason_codes"][0][0] == REASON_MESSAGES["structural_floor"]

    regime_frame = frame.iloc[[1]].copy()
    regime_frame["structural_floor"] = 0.60
    regime_frame["credit_utilisation_regime_change"] = 1.0
    regime_batch = FeatureBatch(regime_frame, vectors[1:2], np.ones(1, dtype=bool), [{}])
    regime = model.predict(regime_batch)
    assert regime["fused"][0] >= 0.60
    assert regime["reason_codes"][0][0] == REASON_MESSAGES["credit_utilisation_regime_change"]
    assert model.predict(batch, ablation="without_semantic")["fused"].shape == (count,)
