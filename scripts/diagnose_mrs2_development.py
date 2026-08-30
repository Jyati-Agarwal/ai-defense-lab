"""Bounded aggregate diagnosis of the terminal MRS2 DEVELOPMENT failure."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim.entity_fidelity_mrs2 as mrs2


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/entity_fidelity_mrs2_development.json"
OUTPUT = ROOT / "results/entity_fidelity_mrs2_development_diagnosis.json"
PRODUCER_FILES = {
    "diagnostic_script_sha256": ROOT / "scripts/diagnose_mrs2_development.py",
    "diagnostic_test_sha256": ROOT / "tests/test_diagnose_mrs2_development.py",
}
EXPECTED_PARENT_FILE_SHA256 = (
    "dddb920b50b89f0fe026c5a4f2a197ad236ec561fb61054190313a6889517453"
)
DIAGNOSIS_SEED = 20260827
AUC_TOLERANCE = 1e-12
FEATURE_PERMUTATION_REPEATS = 16
LABEL_PERMUTATION_REPEATS = 999
LABEL_PERMUTATION_SEED = 20260927
MAX_PARENT_BYTES = 1 << 20
FROZEN_FEATURES = (
    "median_log_amount_percentile",
    "amount_cv",
    "mean_hour_sin",
    "mean_hour_cos",
    "log1p_median_interpurchase_days",
    "log1p_frequency_per_30d",
    "log1p_recency_days",
    "log1p_max_burst_24h",
)
FEATURE_GROUPS = {
    "amount": FROZEN_FEATURES[:2],
    "circadian": FROZEN_FEATURES[2:4],
    "renewal_recency": FROZEN_FEATURES[4:7],
    "burst": FROZEN_FEATURES[7:],
}
STRATA = ("singleton", "same_day_repeat", "multiday_repeat")


@dataclass
class _FoldMaterial:
    fold: int
    training_features: Any
    training_labels: Any
    real_evaluation_features: Any
    synthetic_evaluation_features: Any
    real_evaluation: Any
    synthetic_evaluation: Any
    classifier: Any
    real_predictions: Any
    synthetic_predictions: Any
    public: dict[str, Any]

    @property
    def evaluation_features(self) -> Any:
        import pandas as pd

        return pd.concat(
            (self.real_evaluation_features, self.synthetic_evaluation_features),
            ignore_index=True,
        )

    @property
    def evaluation_labels(self) -> Any:
        import numpy as np

        return np.concatenate(
            (
                np.zeros(len(self.real_evaluation_features), dtype="int8"),
                np.ones(len(self.synthetic_evaluation_features), dtype="int8"),
            )
        )


def _read_regular_snapshot(path: Path, *, maximum_bytes: int) -> tuple[bytes, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("artifact validation requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
    except OSError as error:
        raise FileNotFoundError(f"{path.name} is missing or symbolic") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FileNotFoundError(f"{path.name} is not a regular file")
        if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
            raise ValueError(f"{path.name} size is invalid")
        chunks = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size or len(raw) > maximum_bytes:
            raise ValueError(f"{path.name} changed while reading")
    finally:
        os.close(descriptor)
    return raw, metadata


def _read_parent_envelope(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    raw, metadata = _read_regular_snapshot(path, maximum_bytes=MAX_PARENT_BYTES)
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise ValueError("canonical MRS2 development parent mode must be exactly 0444")
    actual_file_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError("canonical MRS2 development parent file SHA-256 changed")
    try:
        value = json.loads(raw)
        canonical = mrs2._canonical_bytes(value)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("canonical MRS2 development parent is invalid JSON") from error
    if (
        type(value) is not dict
        or raw != canonical
        or value.get("schema_version") != "mrs2"
        or value.get("artifact") != "entity_fidelity_mrs2_development"
        or value.get("study_revision") != "mrs2.0"
        or value.get("status") != "STOP_FAIL"
        or value.get("artifact_sha256") != mrs2._artifact_hash(value)
    ):
        raise ValueError("canonical MRS2 development parent envelope is invalid")
    return value


def _producer_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(
            _read_regular_snapshot(path, maximum_bytes=MAX_PARENT_BYTES)[0]
        ).hexdigest()
        for name, path in PRODUCER_FILES.items()
    }


def _validate_parent_bindings(
    parent: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> Any:
    profile = mrs2.MRS2Profile.from_dict(parent.get("development_profile", {}))
    reference = parent.get("development_reference", {})
    gate = parent.get("gate", {})
    expected_reference = {
        **mrs2._expected_development_loader_reference(preregistration),
        "observation_end": preregistration["completejourney"][
            "observation_end_exclusive"
        ],
        "profile_customer_ids_sha256": profile.source_customer_ids_sha256,
        "profile_transactions_sha256": profile.source_transactions_sha256,
    }
    if (
        parent.get("development_profile_sha256") != profile.sha256
        or parent.get("mrs2_preregistration_sha256")
        != mrs2._file_sha256(mrs2.DEFAULT_PREREGISTRATION)
        or parent.get("development_attempt_sha256")
        != mrs2._file_sha256(mrs2.DEFAULT_DEVELOPMENT_ATTEMPT)
        or parent.get("mrs1_stop_protocol_sha256")
        != mrs2._file_sha256(mrs2.MRS1_STOP_PROTOCOL)
        or parent.get("completejourney") != preregistration.get("completejourney")
        or reference != expected_reference
        or profile.source_partition != "development"
        or profile.source_customer_count != reference.get("households")
        or profile.source_transaction_count != reference.get("baskets")
        or gate.get("gate_passed") is not False
        or not mrs2._gate_evidence_valid(gate)
        or parent.get("implementation_sha256") != mrs2._implementation_hashes()
        or parent.get("runtime_dependency_sha256")
        != mrs2._runtime_dependency_hashes()
        or parent.get("dependency_versions") != mrs2._dependency_versions()
        or parent.get("dependency_lock_sha256")
        != mrs2._file_sha256(ROOT / "requirements.lock")
        or parent.get("development_partition_accessed") is not True
        or parent.get("qualification_accessed") is not False
        or parent.get("qualification_artifact_created") is not False
        or parent.get("uci_sources_accessed") is not False
        or parent.get("retry_permitted") is not False
    ):
        raise ValueError("canonical MRS2 development parent bindings changed")
    if any(
        fold.get("profile_sha256") != profile.sha256
        for seed in gate["seeds"].values()
        for fold in seed["folds"]
    ):
        raise ValueError("canonical MRS2 development fold profile hash changed")
    return profile


def _rng_seed(*parts: object) -> int:
    material = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _auc_d(labels: Any, scores: Any) -> dict[str, float | None]:
    import numpy as np

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    usable = np.isfinite(scores)
    labels = labels[usable]
    scores = scores[usable]
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return {"auc": None, "d": None}
    _, _, _, roc_auc_score = mrs2._dependencies()
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "d": max(auc, 1 - auc)}


def _weighted_fold_d(folds: Sequence[Mapping[str, Any]]) -> float:
    numerator = sum(
        float(fold["d"]) * int(fold["evaluation_real_customers"])
        for fold in folds
    )
    denominator = sum(int(fold["evaluation_real_customers"]) for fold in folds)
    if denominator <= 0:
        raise ValueError("diagnostic folds contain no real evaluation customers")
    return numerator / denominator


def _verify_reproduction(
    reproduced: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(reproduced) != 5 or len(recorded) != 5:
        raise RuntimeError("MRS2 diagnosis requires exactly all five fixed-seed folds")
    recorded_by_fold = {int(fold["fold"]): fold for fold in recorded}
    reproduced_by_fold = {int(fold["fold"]): fold for fold in reproduced}
    if set(recorded_by_fold) != set(range(5)) or set(reproduced_by_fold) != set(
        range(5)
    ):
        raise RuntimeError("MRS2 diagnosis fold membership changed")
    exact_fields = (
        "training_real_customers",
        "evaluation_real_customers",
        "evaluation_synthetic_customers",
        "training_customer_ids_sha256",
        "evaluation_customer_ids_sha256",
        "profile_sha256",
    )
    rows = []
    for fold in range(5):
        expected = recorded_by_fold[fold]
        actual = reproduced_by_fold[fold]
        if any(actual.get(key) != expected.get(key) for key in exact_fields):
            raise RuntimeError(f"MRS2 diagnosis fold {fold} binding changed")
        auc_error = abs(float(actual["auc"]) - float(expected["auc"]))
        d_error = abs(float(actual["d"]) - float(expected["d"]))
        if auc_error > AUC_TOLERANCE or d_error > AUC_TOLERANCE:
            raise RuntimeError(
                f"MRS2 diagnosis fold {fold} did not exactly reproduce recorded AUC"
            )
        rows.append(
            {
                "fold": fold,
                "recorded_auc": float(expected["auc"]),
                "reproduced_auc": float(actual["auc"]),
                "absolute_auc_error": auc_error,
                "evaluation_real_customers": int(actual["evaluation_real_customers"]),
                "evaluation_synthetic_customers": int(
                    actual["evaluation_synthetic_customers"]
                ),
            }
        )
    recorded_d = _weighted_fold_d(recorded)
    reproduced_d = _weighted_fold_d(reproduced)
    if abs(recorded_d - reproduced_d) > AUC_TOLERANCE:
        raise RuntimeError("MRS2 diagnosis fixed-seed weighted D changed")
    return {
        "all_five_folds_reproduced": True,
        "auc_absolute_tolerance": AUC_TOLERANCE,
        "max_absolute_auc_error": max(row["absolute_auc_error"] for row in rows),
        "recorded_fold_weighted_d": recorded_d,
        "reproduced_fold_weighted_d": reproduced_d,
        "folds": rows,
    }


def _reproduce_fixed_seed(
    frame: Any,
    reference: Mapping[str, Any],
    profile: Any,
    recorded: Sequence[Mapping[str, Any]],
) -> tuple[list[_FoldMaterial], dict[str, Any]]:
    import numpy as np
    import pandas as pd

    if DIAGNOSIS_SEED != mrs2.SEED or mrs2.CROSS_FIT_FOLDS != 5:
        raise RuntimeError("frozen MRS2 diagnosis seed or fold count changed")
    if tuple(mrs2.FEATURES) != FROZEN_FEATURES:
        raise RuntimeError("frozen MRS2 diagnostic feature surface changed")
    _, _, _, roc_auc_score = mrs2._dependencies()
    assignments = mrs2._fold_assignments(frame["customer_id"], seed=DIAGNOSIS_SEED)
    source_ids = set(frame["customer_id"].astype(str))
    target_start = mrs2._as_local_wall(reference["observation_start"])
    target_end = mrs2._as_local_wall(reference["observation_end"])
    source_hashes = mrs2._source_sequence_hashes(frame, start=target_start)
    materials: list[_FoldMaterial] = []
    for fold in range(5):
        evaluation_ids = {
            customer_id
            for customer_id, assigned in assignments.items()
            if assigned == fold
        }
        training_ids = set(assignments) - evaluation_ids
        if not training_ids or not evaluation_ids:
            raise ValueError("MRS2 diagnosis reproduced an empty fold")
        real_training = frame.loc[frame["customer_id"].isin(training_ids)].copy()
        real_evaluation = frame.loc[frame["customer_id"].isin(evaluation_ids)].copy()
        synthetic_training = mrs2.generate_synthetic_transactions_mrs2(
            profile,
            customers=len(training_ids),
            seed=DIAGNOSIS_SEED + 10_000 * fold + 101,
            start=target_start,
            end=target_end,
            forbidden_customer_ids=source_ids,
            forbidden_sequence_hashes=source_hashes,
        )
        synthetic_evaluation = mrs2.generate_synthetic_transactions_mrs2(
            profile,
            customers=len(evaluation_ids),
            seed=DIAGNOSIS_SEED + 10_000 * fold + 202,
            start=target_start,
            end=target_end,
            forbidden_customer_ids=source_ids,
            forbidden_sequence_hashes=source_hashes,
        )
        real_transform, synthetic_transform, _ = mrs2.fit_calibrations(
            real_training, synthetic_training
        )
        real_training_features = mrs2.customer_features(real_training, real_transform)
        synthetic_training_features = mrs2.customer_features(
            synthetic_training, synthetic_transform
        )
        training_features = pd.concat(
            (real_training_features, synthetic_training_features), ignore_index=True
        )
        training_labels = np.concatenate(
            (
                np.zeros(len(real_training_features), dtype="int8"),
                np.ones(len(synthetic_training_features), dtype="int8"),
            )
        )
        classifier = mrs2._classifier(DIAGNOSIS_SEED + fold)
        classifier.fit(training_features, training_labels)
        real_evaluation_features = mrs2.customer_features(
            real_evaluation, real_transform
        )
        synthetic_evaluation_features = mrs2.customer_features(
            synthetic_evaluation, synthetic_transform
        )
        real_predictions = classifier.predict_proba(real_evaluation_features)[:, 1]
        synthetic_predictions = classifier.predict_proba(
            synthetic_evaluation_features
        )[:, 1]
        auc = float(
            roc_auc_score(
                np.concatenate(
                    (np.zeros(len(real_predictions)), np.ones(len(synthetic_predictions)))
                ),
                np.concatenate((real_predictions, synthetic_predictions)),
            )
        )
        public = {
            "fold": fold,
            "training_real_customers": len(training_ids),
            "evaluation_real_customers": len(evaluation_ids),
            "evaluation_synthetic_customers": len(synthetic_predictions),
            "training_customer_ids_sha256": mrs2._ids_hash(training_ids),
            "evaluation_customer_ids_sha256": mrs2._ids_hash(evaluation_ids),
            "profile_sha256": profile.sha256,
            "auc": auc,
            "d": max(auc, 1 - auc),
        }
        materials.append(
            _FoldMaterial(
                fold=fold,
                training_features=training_features,
                training_labels=training_labels,
                real_evaluation_features=real_evaluation_features,
                synthetic_evaluation_features=synthetic_evaluation_features,
                real_evaluation=real_evaluation,
                synthetic_evaluation=synthetic_evaluation,
                classifier=classifier,
                real_predictions=real_predictions,
                synthetic_predictions=synthetic_predictions,
                public=public,
            )
        )
    reproduction = _verify_reproduction(
        [material.public for material in materials], recorded
    )
    return materials, reproduction


def _finite_summary(values: Any) -> dict[str, int | float | None]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    quantiles: list[float | None]
    if len(finite):
        quantiles = [float(value) for value in np.quantile(finite, (0.1, 0.5, 0.9))]
    else:
        quantiles = [None, None, None]
    return {
        "customers": int(len(array)),
        "missing_count": int(len(array) - len(finite)),
        "missing_rate": float((len(array) - len(finite)) / len(array))
        if len(array)
        else None,
        "q10": quantiles[0],
        "q50": quantiles[1],
        "q90": quantiles[2],
    }


def _subset_model_d(
    materials: Sequence[_FoldMaterial], columns: Sequence[str]
) -> dict[str, Any]:
    import numpy as np

    rows = []
    for material in materials:
        classifier = mrs2._classifier(DIAGNOSIS_SEED + material.fold)
        classifier.fit(
            material.training_features[list(columns)], material.training_labels
        )
        scores = classifier.predict_proba(
            material.evaluation_features[list(columns)]
        )[:, 1]
        metric = _auc_d(material.evaluation_labels, scores)
        if metric["d"] is None:
            raise RuntimeError("subset diagnostic lost one evaluation domain")
        rows.append(
            {
                "fold": material.fold,
                "evaluation_real_customers": len(material.real_evaluation_features),
                "auc": metric["auc"],
                "d": metric["d"],
            }
        )
    return {"fold_weighted_d": _weighted_fold_d(rows), "folds": rows}


def _permutation_d_drop(
    materials: Sequence[_FoldMaterial], feature: str
) -> dict[str, Any]:
    import numpy as np

    observed = _weighted_fold_d([material.public for material in materials])
    repeat_scores = []
    for repeat in range(FEATURE_PERMUTATION_REPEATS):
        fold_rows = []
        for material in materials:
            evaluation = material.evaluation_features
            permuted = evaluation.copy()
            rng = np.random.default_rng(
                _rng_seed(DIAGNOSIS_SEED, "feature", feature, material.fold, repeat)
            )
            permuted[feature] = rng.permutation(
                evaluation[feature].to_numpy(copy=True)
            )
            metric = _auc_d(
                material.evaluation_labels,
                material.classifier.predict_proba(permuted)[:, 1],
            )
            if metric["d"] is None:
                raise RuntimeError("feature permutation lost one evaluation domain")
            fold_rows.append(
                {
                    "d": metric["d"],
                    "evaluation_real_customers": len(
                        material.real_evaluation_features
                    ),
                }
            )
        repeat_scores.append(_weighted_fold_d(fold_rows))
    permuted_mean = float(np.mean(repeat_scores))
    return {
        "repeats_per_feature": FEATURE_PERMUTATION_REPEATS,
        "observed_fold_weighted_d": observed,
        "permuted_fold_weighted_d_mean": permuted_mean,
        "d_drop": observed - permuted_mean,
    }


def _feature_diagnostics(
    materials: Sequence[_FoldMaterial],
) -> dict[str, dict[str, Any]]:
    import pandas as pd

    result = {}
    for feature in FROZEN_FEATURES:
        real = pd.concat(
            [material.real_evaluation_features[feature] for material in materials]
        ).to_numpy()
        synthetic = pd.concat(
            [material.synthetic_evaluation_features[feature] for material in materials]
        ).to_numpy()
        result[feature] = {
            "real": _finite_summary(real),
            "synthetic": _finite_summary(synthetic),
            "single_feature": _subset_model_d(materials, (feature,)),
            "full_model_permutation": _permutation_d_drop(materials, feature),
        }
    return result


def _customer_strata(frame: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for customer_id, group in frame.groupby("customer_id", sort=True):
        if len(group) == 1:
            stratum = "singleton"
        elif group["ts"].dt.normalize().nunique() == 1:
            stratum = "same_day_repeat"
        else:
            stratum = "multiday_repeat"
        result[str(customer_id)] = stratum
    return result


def _stratum_diagnostics(materials: Sequence[_FoldMaterial]) -> dict[str, Any]:
    import numpy as np

    counts = {stratum: {"real": 0, "synthetic": 0} for stratum in STRATA}
    fold_metrics = {stratum: [] for stratum in STRATA}
    totals = {"real": 0, "synthetic": 0}
    for material in materials:
        fold_scores = {
            stratum: {"real": [], "synthetic": []} for stratum in STRATA
        }
        for domain, frame, features, predictions in (
            (
                "real",
                material.real_evaluation,
                material.real_evaluation_features,
                material.real_predictions,
            ),
            (
                "synthetic",
                material.synthetic_evaluation,
                material.synthetic_evaluation_features,
                material.synthetic_predictions,
            ),
        ):
            strata = _customer_strata(frame)
            if set(strata) != set(map(str, features.index)):
                raise RuntimeError("stratum aggregation lost or added a customer")
            totals[domain] += len(features)
            for customer_id, prediction in zip(
                map(str, features.index), predictions, strict=True
            ):
                stratum = strata[customer_id]
                fold_scores[stratum][domain].append(float(prediction))
                counts[stratum][domain] += 1
        for stratum in STRATA:
            real = fold_scores[stratum]["real"]
            synthetic = fold_scores[stratum]["synthetic"]
            metric = _auc_d(
                np.concatenate((np.zeros(len(real)), np.ones(len(synthetic)))),
                np.asarray(real + synthetic),
            )
            fold_metrics[stratum].append(
                {
                    "fold": material.fold,
                    "real_customers": len(real),
                    "synthetic_customers": len(synthetic),
                    "comparison_pair_weight": len(real) * len(synthetic),
                    "auc": metric["auc"],
                    "d": metric["d"],
                }
            )
    result = {}
    for stratum in STRATA:
        real_count = counts[stratum]["real"]
        synthetic_count = counts[stratum]["synthetic"]
        valid = [row for row in fold_metrics[stratum] if row["d"] is not None]
        weight = sum(row["comparison_pair_weight"] for row in valid)
        result[stratum] = {
            "real_customers": real_count,
            "real_prevalence": real_count / totals["real"] if totals["real"] else None,
            "synthetic_customers": synthetic_count,
            "synthetic_prevalence": synthetic_count / totals["synthetic"]
            if totals["synthetic"]
            else None,
            "within_stratum_auc": sum(
                float(row["auc"]) * row["comparison_pair_weight"] for row in valid
            )
            / weight
            if weight
            else None,
            "within_stratum_d": sum(
                float(row["d"]) * row["comparison_pair_weight"] for row in valid
            )
            / weight
            if weight
            else None,
            "fold_weighting": "real-synthetic comparison pairs; folds missing either domain excluded",
            "folds": fold_metrics[stratum],
        }
    return result


def _label_permutation_null(materials: Sequence[_FoldMaterial]) -> dict[str, Any]:
    import numpy as np

    observed = _weighted_fold_d([material.public for material in materials])
    null_scores = []
    for repeat in range(LABEL_PERMUTATION_REPEATS):
        rows = []
        for material in materials:
            labels = material.evaluation_labels.copy()
            rng = np.random.default_rng(
                _rng_seed(LABEL_PERMUTATION_SEED, material.fold, repeat)
            )
            rng.shuffle(labels)
            scores = np.concatenate(
                (material.real_predictions, material.synthetic_predictions)
            )
            metric = _auc_d(labels, scores)
            if metric["d"] is None:
                raise RuntimeError("label permutation lost one evaluation domain")
            rows.append(
                {
                    "d": metric["d"],
                    "evaluation_real_customers": len(
                        material.real_evaluation_features
                    ),
                }
            )
        null_scores.append(_weighted_fold_d(rows))
    null = np.asarray(null_scores)
    q50, q95, q99 = np.quantile(null, (0.5, 0.95, 0.99))
    return {
        "kind": "fixed evaluation-label permutations of reproduced fold scores",
        "seed": LABEL_PERMUTATION_SEED,
        "repeats": LABEL_PERMUTATION_REPEATS,
        "observed_fold_weighted_d": observed,
        "null_d_q50": float(q50),
        "null_d_q95": float(q95),
        "null_d_q99": float(q99),
        "upper_tail_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
    }


def _count_quantiles(values: Sequence[int]) -> dict[str, float | None]:
    import numpy as np

    if not values:
        return {"q10": None, "q50": None, "q90": None}
    q10, q50, q90 = np.quantile(np.asarray(values, dtype=float), (0.1, 0.5, 0.9))
    return {"q10": float(q10), "q50": float(q50), "q90": float(q90)}


def _volume_mismatch(materials: Sequence[_FoldMaterial]) -> dict[str, Any]:
    real_counts: list[int] = []
    synthetic_counts: list[int] = []
    folds = []
    for material in materials:
        real = [int(value) for value in material.real_evaluation.groupby("customer_id").size()]
        synthetic = [
            int(value)
            for value in material.synthetic_evaluation.groupby("customer_id").size()
        ]
        real_counts.extend(real)
        synthetic_counts.extend(synthetic)
        folds.append(
            {
                "fold": material.fold,
                "real_transactions": sum(real),
                "synthetic_transactions": sum(synthetic),
                "transaction_difference": sum(synthetic) - sum(real),
            }
        )
    real_transactions = sum(real_counts)
    synthetic_transactions = sum(synthetic_counts)
    return {
        "real_customers": len(real_counts),
        "synthetic_customers": len(synthetic_counts),
        "real_transactions": real_transactions,
        "synthetic_transactions": synthetic_transactions,
        "transaction_difference": synthetic_transactions - real_transactions,
        "synthetic_to_real_transaction_ratio": synthetic_transactions
        / real_transactions
        if real_transactions
        else None,
        "real_transactions_per_customer": _count_quantiles(real_counts),
        "synthetic_transactions_per_customer": _count_quantiles(synthetic_counts),
        "folds": folds,
    }


_FORBIDDEN_AGGREGATE_KEYS = frozenset(
    {
        "customer_id",
        "customer_ids",
        "rows",
        "predictions",
        "labels",
        "posteriors",
        "training_features",
        "evaluation_features",
    }
)


def _assert_aggregate_only(value: object, *, key: str = "root") -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            normalized = str(child_key).lower().lstrip("_")
            if normalized in _FORBIDDEN_AGGREGATE_KEYS or (
                "customer_ids" in normalized and not normalized.endswith("sha256")
            ):
                raise ValueError(f"diagnosis contains row-level field {child_key!r}")
            _assert_aggregate_only(child, key=str(child_key))
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            raise ValueError(f"diagnosis contains non-aggregate sequence at {key!r}")
        for child in value:
            _assert_aggregate_only(child, key=key)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"diagnosis contains non-finite number at {key!r}")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"diagnosis contains non-JSON value at {key!r}")


def _write_output(path: Path, value: Mapping[str, Any]) -> None:
    _assert_aggregate_only(value)
    if value.get("diagnostic_only") is not True or value.get(
        "not_release_evidence"
    ) is not True:
        raise ValueError("MRS2 diagnosis must remain diagnostic-only")
    if value.get("artifact_sha256") != mrs2._artifact_hash(value):
        raise ValueError("MRS2 diagnosis internal artifact hash is invalid")
    mrs2._write_artifact(path, value)
    path.chmod(0o444)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError("MRS2 diagnosis output is not immutable mode 0444")


def _bindings(
    parent: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> dict[str, str]:
    reference = parent["development_reference"]
    return {
        **_producer_hashes(),
        "parent_file_sha256": EXPECTED_PARENT_FILE_SHA256,
        "parent_artifact_sha256": parent["artifact_sha256"],
        "mrs2_preregistration_file_sha256": parent[
            "mrs2_preregistration_sha256"
        ],
        "development_attempt_file_sha256": parent["development_attempt_sha256"],
        "mrs1_stop_protocol_file_sha256": parent[
            "mrs1_stop_protocol_sha256"
        ],
        "completejourney_manifest_sha256": reference["manifest_sha256"],
        "development_partition_sha256": reference["partition_sha256"],
        "development_household_ids_sha256": reference["household_ids_sha256"],
        "development_profile_customer_ids_sha256": reference[
            "profile_customer_ids_sha256"
        ],
        "development_profile_transactions_sha256": reference[
            "profile_transactions_sha256"
        ],
        "development_profile_sha256": parent["development_profile_sha256"],
        "source_rds_sha256": reference["source_rds_sha256"],
        "mrs1_r2_preregistration_sha256": preregistration["completejourney"][
            "mrs1_r2_preregistration_sha256"
        ],
    }


def run_diagnosis() -> dict[str, Any]:
    parent = _read_parent_envelope(PARENT, EXPECTED_PARENT_FILE_SHA256)
    preregistration = mrs2.load_mrs2_preregistration()
    profile = _validate_parent_bindings(parent, preregistration)

    # This is the sole data loader call: canonical Complete Journey DEVELOPMENT.
    raw, loaded_reference = mrs2._load_canonical_development(preregistration)
    frame, reference = mrs2._adapt_development(
        raw, loaded_reference, preregistration
    )
    if reference != parent["development_reference"]:
        raise ValueError("canonical MRS2 development source hashes changed")
    recorded = parent["gate"]["seeds"][str(DIAGNOSIS_SEED)]["folds"]
    materials, reproduction = _reproduce_fixed_seed(
        frame, reference, profile, recorded
    )
    value: dict[str, Any] = {
        "schema_version": "mrs2-development-diagnosis-v1",
        "artifact": "entity_fidelity_mrs2_development_diagnosis",
        "parent_status": "STOP_FAIL",
        "diagnostic_only": True,
        "not_release_evidence": True,
        "fixed_seed": DIAGNOSIS_SEED,
        "profile_refit": False,
        "model_candidate_search": False,
        "threshold_tuning": False,
        "qualification_accessed": False,
        "uci_sources_accessed": False,
        "bindings": _bindings(parent, preregistration),
        "reproduction": reproduction,
        "features": _feature_diagnostics(materials),
        "group_only_d": {
            name: _subset_model_d(materials, columns)
            for name, columns in FEATURE_GROUPS.items()
        },
        "strata": _stratum_diagnostics(materials),
        "label_permutation_null": _label_permutation_null(materials),
        "volume_mismatch": _volume_mismatch(materials),
    }
    value["artifact_sha256"] = mrs2._artifact_hash(value)
    _write_output(OUTPUT, value)
    return value


def main() -> None:
    print(json.dumps(run_diagnosis(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
