from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sim.entity_fidelity_v5 as fidelity_v5
from sim.entity_fidelity_v5 import (
    MIN_CHECKERBOARD_CUSTOMERS,
    _assert_aggregate_only,
    _fit_behavior_profile_and_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration_fixture(
    *, singletons: int = 32, same_day: int = 31, multiday: int = 32
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2010-01-01")
    for index in range(singletons):
        rows.append(
            {
                "customer_id": f"singleton-{index}",
                "ts": base + pd.Timedelta(days=index, hours=index % 24),
                "amount": 10.0 + index,
            }
        )
    for index in range(same_day):
        day = base + pd.Timedelta(days=index)
        rows.extend(
            (
                {
                    "customer_id": f"same-{index}",
                    "ts": day + pd.Timedelta(hours=index % 20),
                    "amount": 20.0 + index,
                },
                {
                    "customer_id": f"same-{index}",
                    "ts": day + pd.Timedelta(hours=index % 20 + 1),
                    "amount": 22.0 + index,
                },
            )
        )
    for index in range(multiday):
        day = base + pd.Timedelta(days=index)
        rows.extend(
            (
                {
                    "customer_id": f"multi-{index}",
                    "ts": day + pd.Timedelta(hours=index % 12),
                    "amount": 30.0 + index,
                },
                {
                    "customer_id": f"multi-{index}",
                    "ts": day + pd.Timedelta(days=2, hours=index % 12),
                    "amount": 35.0 + index,
                },
            )
        )
    return pd.DataFrame(rows).sort_values(["customer_id", "ts"], kind="stable")


def test_same_day_atom_has_no_checkerboard_minimum() -> None:
    calibration = _calibration_fixture(same_day=31)

    profile, primitive_rows = _fit_behavior_profile_and_rows(
        calibration, customer_ids_sha256="c" * 64
    )

    assert MIN_CHECKERBOARD_CUSTOMERS == 32
    assert {name: len(rows) for name, rows in primitive_rows.items()} == {
        "singleton": 32,
        "multiday_repeat": 32,
    }
    assert profile.repeat_customer_count == 63
    assert math.isclose(profile.same_day_repeat_probability, 31 / 63)


def test_v4_development_failure_is_immutable_bound_stop_evidence() -> None:
    path = ROOT / "results" / "entity_fidelity_v4_development_failed.json"
    failure = json.loads(path.read_text(encoding="utf-8"))

    assert failure["artifact"] == "entity_fidelity_v4_calibration_development_failure"
    assert failure["status"] == "STOP_FAIL"
    assert failure["protocol_frozen"] is False
    assert failure["threshold_max_d_upper_95"] == 0.65
    assert failure["threshold_changed"] is False
    assert failure["calibration_only"] is True
    assert failure["uci502_transactions_accessed"] is False
    assert failure["official_readiness_run_performed"] is False
    assert failure["diagnostic_transactions_accessed"] is False
    assert failure["sealed_final_transactions_accessed"] is False
    assert failure["failure"]["checkerboard_stratum"] == "same_day_repeat"
    assert failure["failure"]["actual_customers"] == 31
    assert failure["failure"]["required_customers"] == 32
    assert failure["bindings"]["preserved_v3_failure_sha256"] == (
        "9539f959db7724aac7ab529596a67a540820285ed85818c7425068117d0647b8"
    )
    assert _sha256(path) == "81ca82b38ce4194938b4a35be5c5397a4b9e7b67121a249a5057861722d8b312"
    assert path.stat().st_mode & 0o222 == 0
    assert failure["bindings"]["implementation_sha256"]["sim/population.py"] == (
        "a68b006f4f76f0304b80cfa2ea1de57e8ee095c077265da6a6e14daee459d107"
    )


@pytest.mark.parametrize(
    ("singletons", "multiday", "stratum"),
    ((31, 32, "singleton"), (32, 31, "multiday_repeat")),
)
def test_real_checkerboard_strata_below_minimum_fail(
    singletons: int, multiday: int, stratum: str
) -> None:
    calibration = _calibration_fixture(
        singletons=singletons, same_day=31, multiday=multiday
    )

    with pytest.raises(ValueError, match=rf"{stratum} has 31 customers"):
        _fit_behavior_profile_and_rows(calibration, customer_ids_sha256="c" * 64)


@pytest.mark.parametrize("key", ("raw_rows", "scores", "predictions"))
def test_aggregate_allowlist_rejects_sensitive_row_or_score_containers(
    key: str,
) -> None:
    with pytest.raises(ValueError, match=r"not (?:allowed|an allowlisted)"):
        _assert_aggregate_only({key: [{"customer_id": "leak", "value": 0.9}]})


def _valid_exploratory_evidence() -> dict[str, object]:
    folds = [
        {
            "fold": fold,
            "training_real_customers": 1_952,
            "evaluation_real_customers": 488,
            "evaluation_synthetic_customers": 488,
            "training_customer_ids_sha256": "a" * 64,
            "evaluation_customer_ids_sha256": "b" * 64,
            "profile_sha256": "c" * 64,
            "training_repeat_customers": 1_000,
            "training_multiday_repeat_customers": 800,
            "training_same_day_repeat_probability": 0.2,
            "auc": 0.55,
            "d": 0.55,
        }
        for fold in range(fidelity_v5.CROSS_FIT_FOLDS)
    ]
    return {
        "reference": {
            "workbook_sha256": fidelity_v5.UCI352_WORKBOOK_SHA256,
            "partition": "calibration",
            "customer_ids_sha256": fidelity_v5.EXPECTED_PARTITION_HASHES[
                "calibration"
            ],
            "customers": fidelity_v5.UCI352_CALIBRATION_CUSTOMERS,
            "transactions": fidelity_v5.UCI352_CALIBRATION_TRANSACTIONS,
        },
        "seeds": {
            str(seed): {"fold_weighted_d": 0.55, "folds": folds}
            for seed in fidelity_v5.EVALUATION_SEEDS
        },
        "metrics": {
            "orientation_free_max_seed_d": 0.55,
            "bootstrap_max_seed_d_mean": 0.56,
            "bootstrap_max_seed_d_ci_95": [0.52, 0.60],
            "threshold_max_d_upper_95": 0.65,
        },
        "gate_passed": True,
    }


def test_exploratory_validator_rejects_impossible_or_inconsistent_evidence() -> None:
    valid = _valid_exploratory_evidence()
    assert fidelity_v5._exploratory_evidence_valid(valid)

    hostile = []
    candidate = json.loads(json.dumps(valid))
    candidate["metrics"]["bootstrap_max_seed_d_ci_95"] = [0, 0]
    hostile.append(candidate)
    candidate = json.loads(json.dumps(valid))
    candidate["seeds"][str(fidelity_v5.EVALUATION_SEEDS[0])]["folds"][0][
        "d"
    ] = True
    hostile.append(candidate)
    candidate = json.loads(json.dumps(valid))
    candidate["seeds"].pop(str(fidelity_v5.EVALUATION_SEEDS[0]))
    hostile.append(candidate)
    candidate = json.loads(json.dumps(valid))
    candidate["seeds"][str(fidelity_v5.EVALUATION_SEEDS[0])]["folds"].pop()
    hostile.append(candidate)
    candidate = json.loads(json.dumps(valid))
    candidate["seeds"][str(fidelity_v5.EVALUATION_SEEDS[0])][
        "fold_weighted_d"
    ] = 0.6
    hostile.append(candidate)

    assert all(
        not fidelity_v5._exploratory_evidence_valid(candidate)
        for candidate in hostile
    )


def test_crossfit_refits_same_day_probability_inside_training_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _calibration_fixture(singletons=64, same_day=31, multiday=64)
    original_fit = fidelity_v5.fit_behavior_profile_v5
    observed: list[tuple[int, float, float]] = []

    def fitted_training_probability(frame: pd.DataFrame) -> float:
        grouped = frame.groupby("customer_id", sort=False)["ts"]
        repeats = grouped.size()
        repeat_ids = set(repeats[repeats >= 2].index)
        same_day = sum(
            group["ts"].min().normalize() == group["ts"].max().normalize()
            for customer_id, group in frame.groupby("customer_id", sort=False)
            if customer_id in repeat_ids
        )
        return same_day / len(repeat_ids)

    def spy_fit(frame: pd.DataFrame, *, customer_ids_sha256: str):
        profile = original_fit(frame, customer_ids_sha256=customer_ids_sha256)
        observed.append(
            (
                frame["customer_id"].nunique(),
                profile.same_day_repeat_probability,
                fitted_training_probability(frame),
            )
        )
        return profile

    def fake_synthetic_group(
        _profile: object, *, customers: int, seed: int, prefix: str
    ) -> pd.DataFrame:
        del seed
        return pd.DataFrame(
            {"customer_id": [f"{prefix}{index}" for index in range(customers)]}
        )

    def fake_features(frame: pd.DataFrame, _transform: object) -> pd.DataFrame:
        customer_ids = sorted(set(frame["customer_id"].astype(str)))
        return pd.DataFrame(
            {"feature": np.arange(len(customer_ids), dtype=float)},
            index=customer_ids,
        )

    class FakeClassifier:
        def fit(self, _features: pd.DataFrame, _labels: np.ndarray) -> "FakeClassifier":
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.full((len(features), 2), 0.5)

    monkeypatch.setattr(fidelity_v5, "fit_behavior_profile_v5", spy_fit)
    monkeypatch.setattr(fidelity_v5, "_synthetic_group", fake_synthetic_group)
    monkeypatch.setattr(
        fidelity_v5, "fit_calibrations", lambda *_args: (object(), object(), {})
    )
    monkeypatch.setattr(fidelity_v5, "customer_features", fake_features)
    monkeypatch.setattr(fidelity_v5, "_classifier", lambda _seed: FakeClassifier())

    folds = fidelity_v5._crossfit_seed(calibration, seed=17, fold_numbers=(0, 1))

    assert len(observed) == len(folds) == 2
    for fold, (training_customers, fitted_q, expected_q) in zip(
        folds, observed, strict=True
    ):
        assert training_customers < calibration["customer_id"].nunique()
        assert math.isclose(fitted_q, expected_q)
        assert math.isclose(
            fold["training_same_day_repeat_probability"], expected_q
        )


def _exploratory_artifact(*, passed: bool = True) -> dict[str, object]:
    result = {
        "schema_version": 5,
        "artifact": "entity_fidelity_v5_calibration_exploratory",
        "protocol_sha256": "frozen",
        "calibration_only": True,
        "uci352_diagnostic_transactions_materialized_or_used": False,
        "uci502_transactions_accessed": False,
        "uci352_prior_holdout_transactions_materialized_or_used": False,
        **_valid_exploratory_evidence(),
    }
    if not passed:
        result["metrics"]["bootstrap_max_seed_d_mean"] = 0.68
        result["metrics"]["bootstrap_max_seed_d_ci_95"] = [0.66, 0.70]
        result["gate_passed"] = False
    return result


def _patch_canonical_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Path]:
    paths = {
        "protocol_path": tmp_path / "protocol.json",
        "exploratory_path": tmp_path / "exploratory.json",
        "receipt_path": tmp_path / "receipt.json",
        "final_path": tmp_path / "final.json",
        "final_reference": tmp_path / "uci502.xlsx",
    }
    monkeypatch.setattr(fidelity_v5, "DEFAULT_PROTOCOL", paths["protocol_path"])
    monkeypatch.setattr(fidelity_v5, "DEFAULT_EXPLORATORY", paths["exploratory_path"])
    monkeypatch.setattr(fidelity_v5, "DEFAULT_SEAL_RECEIPT", paths["receipt_path"])
    monkeypatch.setattr(fidelity_v5, "DEFAULT_OUTPUT", paths["final_path"])
    monkeypatch.setattr(
        fidelity_v5, "DEFAULT_FINAL_REFERENCE", paths["final_reference"]
    )
    return paths


def _write_valid_one_shot_artifacts(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, object], dict[str, object], dict[str, object]]:
    paths = {
        "protocol_path": tmp_path / "protocol.json",
        "exploratory_path": tmp_path / "exploratory.json",
        "receipt_path": tmp_path / "receipt.json",
        "final_path": tmp_path / "final.json",
    }
    exploratory = _exploratory_artifact()
    paths["exploratory_path"].write_text(
        json.dumps(exploratory), encoding="utf-8"
    )
    profile = fidelity_v5.fit_behavior_profile_v5(
        _calibration_fixture(), customer_ids_sha256="c" * 64
    )
    protocol = {
        "protocol_sha256": "frozen",
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    receipt = {
        "schema_version": 5,
        "artifact": "entity_fidelity_v5_one_shot_final_receipt",
        "protocol_sha256": "frozen",
        "status": "consumed",
        "confirmation_sha256": hashlib.sha256(
            b"APPROVED_SINGLE_FINAL_SCORE:frozen"
        ).hexdigest(),
        "exploratory_artifact_sha256": _sha256(paths["exploratory_path"]),
        "recomputed_exploratory_sha256": fidelity_v5._canonical_hash(exploratory),
        "behavior_profile_sha256": profile.sha256,
        "final_dataset_id": fidelity_v5.UCI502_DATASET_ID,
        "final_source_url": fidelity_v5.UCI502_SOURCE_URL,
        "final_archive_sha256": fidelity_v5.UCI502_ARCHIVE_SHA256,
        "final_workbook_sha256": fidelity_v5.UCI502_WORKBOOK_SHA256,
        "final_manifest_sha256": fidelity_v5.UCI502_MANIFEST_SHA256,
        "final_license": fidelity_v5.UCI502_LICENSE,
        "final_downloader_sha256": _sha256(
            ROOT / "scripts" / "download_retail_ii_reference.py"
        ),
        "final_sheet": fidelity_v5.UCI502_SHEET,
        "final_fixed_window": (
            f"[{fidelity_v5.UCI502_WINDOW_START}, "
            f"{fidelity_v5.UCI502_WINDOW_END})"
        ),
        "final_split_assignment": fidelity_v5.UCI502_SPLIT_ASSIGNMENT,
        "excluded_uci352_calibration_customer_ids_sha256": (
            fidelity_v5.EXPECTED_PARTITION_HASHES["calibration"]
        ),
        "receipt_before_final_transaction_sheet_parse_or_score": True,
    }
    receipt["receipt_sha256"] = fidelity_v5._canonical_hash(receipt)
    score_customers = fidelity_v5.UCI502_MIN_SPLIT_CUSTOMERS
    final = {
        "schema_version": 5,
        "artifact": "entity_fidelity_v5_uci502_one_shot_final",
        "protocol_sha256": "frozen",
        "protocol": {
            "protocol_sha256": "frozen",
            "evaluation_partition": "uci502_score_half",
            "one_shot_final_scored": True,
            "development_gate_recomputed_and_exactly_matched": True,
            "orientation_free_statistic": "D=max(AUC,1-AUC)",
            "bootstrap_repeats": fidelity_v5.BOOTSTRAP_REPEATS,
            "shared_real_bootstrap_indices_across_seeds": True,
            "independent_synthetic_bootstrap_indices_across_seeds": True,
        },
        "bindings": {
            "exploratory_artifact_sha256": _sha256(paths["exploratory_path"]),
            "final_receipt_sha256": receipt["receipt_sha256"],
            "behavior_profile_sha256": profile.sha256,
        },
        "behavior_profile": profile.as_dict(),
        "reference": {
            "development_training": {
                "dataset_id": fidelity_v5.UCI352_DATASET_ID,
                "source_url": fidelity_v5.UCI352_SOURCE_URL,
                "workbook_sha256": fidelity_v5.UCI352_WORKBOOK_SHA256,
                "manifest_sha256": fidelity_v5.UCI352_MANIFEST_SHA256,
                "license": fidelity_v5.UCI352_LICENSE,
                "downloader_sha256": _sha256(
                    ROOT / "scripts" / "download_retail_reference.py"
                ),
                "calibration_customer_ids_sha256": (
                    fidelity_v5.EXPECTED_PARTITION_HASHES["calibration"]
                ),
                "calibration_customers": fidelity_v5.UCI352_CALIBRATION_CUSTOMERS,
                "full_workbook_customer_ids_scanned_for_routing": True,
                "noncalibration_transaction_values_materialized_or_used": False,
                "uci352_diagnostic_transactions_materialized_or_used": False,
                "uci352_prior_holdout_transactions_materialized_or_used": False,
            },
            "one_shot_final": {
                "dataset_id": fidelity_v5.UCI502_DATASET_ID,
                "source_url": fidelity_v5.UCI502_SOURCE_URL,
                "archive_sha256": fidelity_v5.UCI502_ARCHIVE_SHA256,
                "workbook_sha256": fidelity_v5.UCI502_WORKBOOK_SHA256,
                "manifest_sha256": fidelity_v5.UCI502_MANIFEST_SHA256,
                "license": fidelity_v5.UCI502_LICENSE,
                "downloader_sha256": _sha256(
                    ROOT / "scripts" / "download_retail_ii_reference.py"
                ),
                "sheet": fidelity_v5.UCI502_SHEET,
                "fixed_window": (
                    f"[{fidelity_v5.UCI502_WINDOW_START}, "
                    f"{fidelity_v5.UCI502_WINDOW_END})"
                ),
                "split_assignment": fidelity_v5.UCI502_SPLIT_ASSIGNMENT,
                "excluded_uci352_calibration_customer_ids_sha256": (
                    fidelity_v5.EXPECTED_PARTITION_HASHES["calibration"]
                ),
                "transaction_time_disjoint": True,
                "customer_disjoint_from_profile_and_classifier_training": True,
                "same_retailer": True,
                "source_independent": False,
                "temporal_direction": "historical_before_training_period",
                "prospective_validation": False,
                "transaction_sheet_parsed_or_scored_before_receipt": False,
                "first_worksheet_opened_after_receipt": True,
                "second_worksheet_opened": False,
                "nonselected_transactions_materialized": False,
                "eligible_customers": 2 * score_customers,
                "eligible_customer_ids_sha256": "f" * 64,
                "transform_customers": score_customers,
                "score_customers": score_customers,
                "transform_customer_ids_sha256": "d" * 64,
                "score_customer_ids_sha256": "e" * 64,
                "transform_amount_ecdf_fit_on_transform_only": True,
                "transform_used_for_profile_generator_classifier_threshold_or_abort": False,
                "score_half_is_only_real_final_evaluation": True,
            },
        },
        "seeds": {
            str(seed): {
                "auc": 0.55,
                "d": 0.55,
                "real_customers": score_customers,
                "synthetic_customers": score_customers,
            }
            for seed in fidelity_v5.EVALUATION_SEEDS
        },
        "metrics": {
            "orientation_free_max_seed_d": 0.55,
            "bootstrap_max_seed_d_mean": 0.56,
            "bootstrap_max_seed_d_ci_95": [0.52, 0.60],
            "threshold_max_d_upper_95": 0.65,
            "gate_passed": True,
        },
    }
    paths["receipt_path"].write_text(json.dumps(receipt), encoding="utf-8")
    paths["final_path"].write_text(json.dumps(final), encoding="utf-8")
    return paths, protocol, receipt, final


def test_complete_internally_consistent_exploratory_fail_is_rejected(
    tmp_path: Path,
) -> None:
    failed = _exploratory_artifact(passed=False)
    path = tmp_path / "failed-exploratory.json"
    path.write_text(json.dumps(failed), encoding="utf-8")

    assert fidelity_v5._exploratory_evidence_valid(failed)
    with pytest.raises(ValueError, match="exploratory gate"):
        fidelity_v5._load_passed_exploratory(
            path, {"protocol_sha256": "frozen"}
        )


def test_exact_exploratory_recompute_mismatch_blocks_before_receipt_or_uci502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_canonical_paths(monkeypatch, tmp_path)
    stored = _exploratory_artifact()
    paths["exploratory_path"].write_text(json.dumps(stored), encoding="utf-8")
    recomputed = json.loads(json.dumps(stored))
    recomputed["metrics"]["bootstrap_max_seed_d_mean"] = 0.57
    protocol = {"protocol_sha256": "frozen"}
    calibration = _calibration_fixture()
    monkeypatch.setattr(
        fidelity_v5, "load_frozen_v5_protocol", lambda _path: protocol
    )
    monkeypatch.setattr(
        fidelity_v5,
        "load_retail_partition",
        lambda _path, partition: (calibration, {})
        if partition == "calibration"
        else pytest.fail("no UCI352 final partition exists"),
    )
    monkeypatch.setattr(
        fidelity_v5,
        "_compute_calibration_exploratory",
        lambda *_args: recomputed,
    )
    monkeypatch.setattr(
        fidelity_v5,
        "load_retail_ii_final",
        lambda *_args, **_kwargs: pytest.fail("UCI502 must remain unopened"),
    )

    with pytest.raises(ValueError, match="bytes differ from exact frozen recomputation"):
        fidelity_v5.run_entity_fidelity_v5_final(
            protocol_path=paths["protocol_path"],
            exploratory_path=paths["exploratory_path"],
            one_shot_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:frozen",
        )
    assert not paths["receipt_path"].exists()
    assert not paths["final_path"].exists()


def test_one_shot_final_rejects_noncanonical_paths_and_wrong_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_canonical_paths(monkeypatch, tmp_path)
    protocol = {"protocol_sha256": "frozen"}
    monkeypatch.setattr(
        fidelity_v5, "load_frozen_v5_protocol", lambda _path: protocol
    )
    monkeypatch.setattr(
        fidelity_v5,
        "_recompute_and_match_exploratory",
        lambda *_args: pytest.fail("exploratory recomputation must not start"),
    )
    monkeypatch.setattr(
        fidelity_v5,
        "load_retail_ii_final",
        lambda *_args, **_kwargs: pytest.fail("UCI502 must remain unopened"),
    )

    for protocol_path, exploratory_path in (
        (tmp_path / "alternate-protocol.json", paths["exploratory_path"]),
        (paths["protocol_path"], tmp_path / "alternate-exploratory.json"),
    ):
        with pytest.raises(ValueError, match="paths are canonical"):
            fidelity_v5.run_entity_fidelity_v5_final(
                protocol_path=protocol_path,
                exploratory_path=exploratory_path,
                one_shot_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:frozen",
            )

    with pytest.raises(ValueError, match="protocol-hash-bound approval"):
        fidelity_v5.run_entity_fidelity_v5_final(
            protocol_path=paths["protocol_path"],
            exploratory_path=paths["exploratory_path"],
            one_shot_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:wrong",
        )
    assert not paths["receipt_path"].exists()


def test_receipt_precedes_uci502_loader_and_failure_burns_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_canonical_paths(monkeypatch, tmp_path)
    calibration = _calibration_fixture()
    profile = fidelity_v5.fit_behavior_profile_v5(
        calibration, customer_ids_sha256="c" * 64
    )
    exploratory = _exploratory_artifact()
    paths["exploratory_path"].write_text(json.dumps(exploratory), encoding="utf-8")
    protocol = {
        "protocol_sha256": "frozen",
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    calls: list[Path] = []

    def fail_at_uci502(path: Path, *, excluded_customer_ids: set[str]):
        calls.append(path)
        assert path == paths["final_reference"]
        assert excluded_customer_ids == set(calibration["customer_id"].astype(str))
        assert paths["receipt_path"].exists()
        assert not paths["final_path"].exists()
        receipt = json.loads(paths["receipt_path"].read_text(encoding="utf-8"))
        assert receipt["status"] == "consumed"
        assert receipt["receipt_before_final_transaction_sheet_parse_or_score"] is True
        raise RuntimeError("UCI502 boundary reached after receipt")

    monkeypatch.setattr(
        fidelity_v5, "load_frozen_v5_protocol", lambda _path: protocol
    )
    monkeypatch.setattr(
        fidelity_v5,
        "_recompute_and_match_exploratory",
        lambda *_args: (calibration, {}, exploratory),
    )
    monkeypatch.setattr(
        fidelity_v5, "fit_behavior_profile_v5", lambda *_args, **_kwargs: profile
    )
    monkeypatch.setattr(fidelity_v5, "load_retail_ii_final", fail_at_uci502)

    with pytest.raises(RuntimeError, match="after receipt"):
        fidelity_v5.run_entity_fidelity_v5_final(
            protocol_path=paths["protocol_path"],
            exploratory_path=paths["exploratory_path"],
            one_shot_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:frozen",
        )
    assert calls == [paths["final_reference"]]

    with pytest.raises(FileExistsError, match="receipt already exists"):
        fidelity_v5.run_entity_fidelity_v5_final(
            protocol_path=paths["protocol_path"],
            exploratory_path=paths["exploratory_path"],
            one_shot_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:frozen",
        )
    assert calls == [paths["final_reference"]]


def test_verify_is_false_for_absent_and_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fidelity_v5,
        "load_frozen_v5_protocol",
        lambda _path: {"protocol_sha256": "frozen"},
    )
    paths = {
        "protocol_path": tmp_path / "protocol.json",
        "exploratory_path": tmp_path / "exploratory.json",
        "receipt_path": tmp_path / "receipt.json",
        "final_path": tmp_path / "final.json",
    }

    absent = fidelity_v5.verify_v5_artifacts(**paths)
    assert absent["verified"] is False
    assert absent["checks"]["exploratory_exists"] is False
    assert absent["checks"]["one_shot_artifacts_complete"] is False

    paths["exploratory_path"].write_text("{}\n", encoding="utf-8")
    partial = fidelity_v5.verify_v5_artifacts(**paths)
    assert partial["verified"] is False
    assert partial["checks"]["exploratory_exists"] is True
    assert partial["checks"]["exploratory_valid"] is False


def test_positive_new_schema_verifier_has_no_readiness_or_legacy_sealed_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, protocol, receipt, final = _write_valid_one_shot_artifacts(tmp_path)
    monkeypatch.setattr(
        fidelity_v5, "load_frozen_v5_protocol", lambda _path: protocol
    )

    verification = fidelity_v5.verify_v5_artifacts(**paths)
    assert verification["verified"] is True
    assert not hasattr(fidelity_v5, "DEFAULT_READINESS")
    assert not hasattr(fidelity_v5, "run_temporal_readiness")
    legacy_fields = {
        "readiness_artifact_sha256",
        "readiness_gate_passed",
        "sealed_partition_customer_ids_sha256",
        "sealed_receipt_sha256",
        "sealed_final_customer_ids_sha256",
        "sealed_final_customers",
        "sealed_final_scored",
    }
    serialized = json.dumps({"receipt": receipt, "final": final}, sort_keys=True)
    assert all(f'"{field}"' not in serialized for field in legacy_fields)
    assert final["artifact"] == "entity_fidelity_v5_uci502_one_shot_final"
    assert final["protocol"]["evaluation_partition"] == "uci502_score_half"
    assert (
        final["reference"]["development_training"][
            "uci352_prior_holdout_transactions_materialized_or_used"
        ]
        is False
    )


@pytest.mark.parametrize(
    "corruption", ("confirmation", "source", "split", "seeds", "final_gate")
)
def test_verifier_rejects_new_schema_corruption(
    corruption: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, protocol, receipt, final = _write_valid_one_shot_artifacts(tmp_path)
    monkeypatch.setattr(
        fidelity_v5, "load_frozen_v5_protocol", lambda _path: protocol
    )

    if corruption == "confirmation":
        receipt["confirmation_sha256"] = "0" * 64
    elif corruption == "source":
        final["reference"]["one_shot_final"]["workbook_sha256"] = "0" * 64
    elif corruption == "split":
        receipt["final_split_assignment"] = "tampered split"
    elif corruption == "seeds":
        final["seeds"].pop(str(fidelity_v5.EVALUATION_SEEDS[0]))
    else:
        final["metrics"]["gate_passed"] = False
    if corruption in {"confirmation", "split"}:
        receipt["receipt_sha256"] = fidelity_v5._canonical_hash(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
        final["bindings"]["final_receipt_sha256"] = receipt["receipt_sha256"]
    paths["receipt_path"].write_text(json.dumps(receipt), encoding="utf-8")
    paths["final_path"].write_text(json.dumps(final), encoding="utf-8")

    verification = fidelity_v5.verify_v5_artifacts(**paths)
    assert verification["verified"] is False
    if corruption in {"confirmation", "split"}:
        assert verification["checks"]["receipt_valid"] is False
    else:
        assert verification["checks"]["final_valid"] is False
