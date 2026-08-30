from __future__ import annotations

import hashlib
import inspect
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sim.entity_fidelity_mrs2 as mrs2
from scripts import diagnose_mrs2_development as diagnosis


def _parent_envelope() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "mrs2",
        "artifact": "entity_fidelity_mrs2_development",
        "study_revision": "mrs2.0",
        "status": "STOP_FAIL",
    }
    value["artifact_sha256"] = mrs2._artifact_hash(value)
    return value


def _write_read_only(path: Path, value: dict[str, object]) -> str:
    raw = mrs2._canonical_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


def test_parent_hash_mode_and_internal_hash_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "parent.json"
    value = _parent_envelope()
    digest = _write_read_only(path, value)

    assert diagnosis._read_parent_envelope(path, digest) == value

    with pytest.raises(ValueError, match="file SHA-256 changed"):
        diagnosis._read_parent_envelope(path, "0" * 64)

    path.chmod(0o644)
    with pytest.raises(ValueError, match="exactly 0444"):
        diagnosis._read_parent_envelope(path, digest)

    value["artifact_sha256"] = "0" * 64
    changed_digest = _write_read_only(path, value)
    with pytest.raises(ValueError, match="envelope is invalid"):
        diagnosis._read_parent_envelope(path, changed_digest)


def test_parent_read_uses_one_fd_snapshot_during_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "parent.json"
    replacement_path = tmp_path / "replacement.json"
    original = _parent_envelope()
    expected = _write_read_only(path, original)
    replacement = {**_parent_envelope(), "marker": "replacement"}
    replacement["artifact_sha256"] = mrs2._artifact_hash(replacement)
    _write_read_only(replacement_path, replacement)
    real_open = os.open
    calls = 0

    def swap_after_open(target: Path, flags: int) -> int:
        nonlocal calls
        calls += 1
        descriptor = real_open(target, flags)
        replacement_path.replace(path)
        return descriptor

    monkeypatch.setattr(diagnosis.os, "open", swap_after_open)

    assert diagnosis._read_parent_envelope(path, expected) == original
    assert calls == 1


def test_producer_bindings_hash_exact_regular_non_symlink_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "diagnosis.py"
    test = tmp_path / "test_diagnosis.py"
    script.write_bytes(b"producer revision\n")
    test.write_bytes(b"producer regression\n")
    monkeypatch.setattr(
        diagnosis,
        "PRODUCER_FILES",
        {
            "diagnostic_script_sha256": script,
            "diagnostic_test_sha256": test,
        },
    )

    assert diagnosis._producer_hashes() == {
        "diagnostic_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "diagnostic_test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
    }

    script.unlink()
    script.symlink_to(test)
    with pytest.raises(FileNotFoundError, match="missing or symbolic"):
        diagnosis._producer_hashes()


def _fold(fold: int, auc: float) -> dict[str, object]:
    return {
        "fold": fold,
        "training_real_customers": 80 + fold,
        "evaluation_real_customers": 20,
        "evaluation_synthetic_customers": 20,
        "training_customer_ids_sha256": f"{fold + 1:064x}",
        "evaluation_customer_ids_sha256": f"{fold + 11:064x}",
        "profile_sha256": "f" * 64,
        "auc": auc,
        "d": max(auc, 1 - auc),
    }


def test_exact_reproduction_seam_requires_all_five_auc_matches() -> None:
    recorded = [_fold(fold, 0.71 + fold / 100) for fold in range(5)]
    reproduced = [dict(row) for row in recorded]
    reproduced[2]["auc"] = float(reproduced[2]["auc"]) + 5e-13
    reproduced[2]["d"] = float(reproduced[2]["d"]) + 5e-13

    evidence = diagnosis._verify_reproduction(reproduced, recorded)

    assert evidence["all_five_folds_reproduced"] is True
    assert evidence["max_absolute_auc_error"] <= 1e-12

    mismatched = [dict(row) for row in recorded]
    mismatched[2]["auc"] = float(mismatched[2]["auc"]) + 2e-12
    mismatched[2]["d"] = float(mismatched[2]["d"]) + 2e-12
    with pytest.raises(RuntimeError, match="exactly reproduce recorded AUC"):
        diagnosis._verify_reproduction(mismatched, recorded)
    with pytest.raises(RuntimeError, match="exactly all five"):
        diagnosis._verify_reproduction(reproduced[:-1], recorded)


def _evaluation_frame(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": f"{prefix}_singleton",
                "ts": pd.Timestamp("2026-01-01 10:00:00"),
                "amount": 100,
            },
            {
                "customer_id": f"{prefix}_same",
                "ts": pd.Timestamp("2026-01-02 10:00:00"),
                "amount": 100,
            },
            {
                "customer_id": f"{prefix}_same",
                "ts": pd.Timestamp("2026-01-02 11:00:00"),
                "amount": 110,
            },
            {
                "customer_id": f"{prefix}_multi",
                "ts": pd.Timestamp("2026-01-03 10:00:00"),
                "amount": 120,
            },
            {
                "customer_id": f"{prefix}_multi",
                "ts": pd.Timestamp("2026-01-05 10:00:00"),
                "amount": 130,
            },
        ]
    )


def _features(prefix: str, values: list[float]) -> pd.DataFrame:
    index = [f"{prefix}_singleton", f"{prefix}_same", f"{prefix}_multi"]
    return pd.DataFrame(
        {
            feature: values if feature == diagnosis.FROZEN_FEATURES[4] else [0.1, 0.2, 0.3]
            for feature in diagnosis.FROZEN_FEATURES
        },
        index=index,
    )


def _material() -> diagnosis._FoldMaterial:
    real_features = _features("real", [math.nan, 1.0, 2.0])
    synthetic_features = _features("synthetic", [math.nan, 3.0, 4.0])
    return diagnosis._FoldMaterial(
        fold=0,
        training_features=pd.concat((real_features, synthetic_features), ignore_index=True),
        training_labels=np.asarray([0, 0, 0, 1, 1, 1], dtype="int8"),
        real_evaluation_features=real_features,
        synthetic_evaluation_features=synthetic_features,
        real_evaluation=_evaluation_frame("real"),
        synthetic_evaluation=_evaluation_frame("synthetic"),
        classifier=None,
        real_predictions=np.asarray([0.1, 0.2, 0.3]),
        synthetic_predictions=np.asarray([0.9, 0.8, 0.7]),
        public={
            "fold": 0,
            "evaluation_real_customers": 3,
            "d": 1.0,
        },
    )


def test_nan_summaries_and_strata_are_aggregate_and_finite() -> None:
    summary = diagnosis._finite_summary([math.nan, 1.0, 2.0, math.inf])

    assert summary == {
        "customers": 4,
        "missing_count": 2,
        "missing_rate": 0.5,
        "q10": pytest.approx(1.1),
        "q50": pytest.approx(1.5),
        "q90": pytest.approx(1.9),
    }
    assert diagnosis._finite_summary([math.nan])["q50"] is None

    material = _material()
    assert set(diagnosis._customer_strata(material.real_evaluation).values()) == set(
        diagnosis.STRATA
    )
    strata = diagnosis._stratum_diagnostics([material])
    for value in strata.values():
        assert value["real_customers"] == 1
        assert value["synthetic_customers"] == 1
        assert value["real_prevalence"] == pytest.approx(1 / 3)
        assert value["synthetic_prevalence"] == pytest.approx(1 / 3)
        assert value["within_stratum_d"] == 1.0

    volume = diagnosis._volume_mismatch([material])
    assert volume["real_transactions"] == 5
    assert volume["synthetic_transactions"] == 5
    diagnosis._assert_aggregate_only({"summary": summary, "strata": strata})


def _offset_material(
    fold: int, *, real_singletons: int, synthetic_singletons: int, score: float
) -> diagnosis._FoldMaterial:
    def domain(prefix: str, singletons: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        ids = [f"{prefix}_{fold}_{index:02d}" for index in range(10)]
        rows = []
        for index, customer_id in enumerate(ids):
            rows.append(
                {
                    "customer_id": customer_id,
                    "ts": pd.Timestamp("2026-01-01 10:00:00"),
                    "amount": 100,
                }
            )
            if index >= singletons:
                rows.append(
                    {
                        "customer_id": customer_id,
                        "ts": pd.Timestamp("2026-01-01 11:00:00"),
                        "amount": 110,
                    }
                )
        return pd.DataFrame(rows), pd.DataFrame(index=ids)

    real, real_features = domain("real", real_singletons)
    synthetic, synthetic_features = domain("synthetic", synthetic_singletons)
    return diagnosis._FoldMaterial(
        fold=fold,
        training_features=pd.DataFrame(),
        training_labels=np.asarray([], dtype="int8"),
        real_evaluation_features=real_features,
        synthetic_evaluation_features=synthetic_features,
        real_evaluation=real,
        synthetic_evaluation=synthetic,
        classifier=None,
        real_predictions=np.full(10, score),
        synthetic_predictions=np.full(10, score),
        public={"fold": fold, "evaluation_real_customers": 10, "d": 0.5},
    )


def test_stratum_d_is_fold_local_despite_score_offsets_and_opposite_prevalence() -> None:
    materials = [
        _offset_material(0, real_singletons=9, synthetic_singletons=1, score=0.8),
        _offset_material(1, real_singletons=1, synthetic_singletons=9, score=0.2),
    ]
    pooled_old_metric = diagnosis._auc_d(
        np.asarray([0] * 9 + [1] + [0] + [1] * 9),
        np.asarray([0.8] * 10 + [0.2] * 10),
    )

    assert pooled_old_metric["d"] > 0.8
    singleton = diagnosis._stratum_diagnostics(materials)["singleton"]
    assert [row["d"] for row in singleton["folds"]] == [0.5, 0.5]
    assert singleton["within_stratum_auc"] == 0.5
    assert singleton["within_stratum_d"] == 0.5


def test_output_is_canonical_aggregate_only_and_exclusive(tmp_path: Path) -> None:
    assert tuple(inspect.signature(diagnosis.run_diagnosis).parameters) == ()
    value: dict[str, object] = {
        "artifact": "entity_fidelity_mrs2_development_diagnosis",
        "diagnostic_only": True,
        "not_release_evidence": True,
        "folds": [{"fold": fold, "auc": 0.5} for fold in range(5)],
    }
    value["artifact_sha256"] = mrs2._artifact_hash(value)
    path = tmp_path / "diagnosis.json"

    diagnosis._write_output(path, value)

    assert path.stat().st_mode & 0o777 == 0o444
    assert path.read_bytes() == mrs2._canonical_bytes(value)
    with pytest.raises(FileExistsError):
        diagnosis._write_output(path, value)
    with pytest.raises(ValueError, match="row-level field"):
        diagnosis._assert_aggregate_only({"customer_ids": ["raw-id"]})
