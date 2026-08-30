from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sim.entity_fidelity_v8 as fidelity_v8


def calibration_fixture(*, tail_count: int = 256) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2010-01-01")
    for index in range(32):
        rows.append(
            {
                "customer_id": f"singleton-{index}",
                "ts": base + pd.Timedelta(days=index, hours=index % 24),
                "amount": 10.0 + index,
            }
        )
    for index in range(31):
        day = base + pd.Timedelta(days=index)
        for hour in (0, 1):
            rows.append(
                {
                    "customer_id": f"same-{index}",
                    "ts": day + pd.Timedelta(hours=hour),
                    "amount": 20.0 + index + hour,
                }
            )
    for index in range(32):
        count = tail_count if index == 31 else 2
        for event in range(count):
            rows.append(
                {
                    "customer_id": f"multi-{index}",
                    "ts": base
                    + pd.Timedelta(days=index)
                    + pd.Timedelta(days=2 * event / (count - 1)),
                    "amount": 30.0 + index + event / 1_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["customer_id", "ts"], kind="stable")


def _futility(*, passed: bool) -> dict[str, object]:
    q_025 = 0.64 if passed else 0.66
    return {
        "seed": fidelity_v8.FUTILITY_SEED,
        "folds": list(fidelity_v8.FUTILITY_FOLDS),
        "real_customers": 12,
        "synthetic_customers": 12,
        "fold_weighted_d": 0.6,
        "bootstrap_repeats": fidelity_v8.FUTILITY_BOOTSTRAP_REPEATS,
        "bootstrap_rng_seed": fidelity_v8.FUTILITY_BOOTSTRAP_SEED,
        "bootstrap_quantile": fidelity_v8.FUTILITY_QUANTILE,
        "bootstrap_d_mean": 0.67,
        "bootstrap_d_q_025": q_025,
        "threshold_max_d_upper_95": fidelity_v8.GATE_MAX_D_UPPER_95,
        "passed_to_full_gate": passed,
    }


def test_v8_fit_persists_exact_repeat_pmf_tail_and_rejects_overflow() -> None:
    profile = fidelity_v8.fit_behavior_profile_v8(
        calibration_fixture(), customer_ids_sha256="c" * 64
    )

    assert len(profile.repeat_invoice_count_pmf) == 257
    assert profile.repeat_invoice_count_pmf[:2] == (0.0, 0.0)
    assert profile.repeat_invoice_count_pmf[2] == 62 / 63
    assert profile.repeat_invoice_count_pmf[256] == 1 / 63
    assert sum(profile.repeat_invoice_count_pmf) == pytest.approx(1.0)
    assert profile.strata[1].primitive_quantiles[0][0] == 2
    assert profile.strata[1].primitive_quantiles[0][-1] == 256
    assert len(profile.strata[1].cell_masses) == 69_984
    tensor = np.asarray(profile.strata[1].cell_masses).reshape(
        fidelity_v8.REPEAT_CHECKERBOARD_SHAPE
    )
    for coordinate, bins in enumerate(fidelity_v8.REPEAT_CHECKERBOARD_SHAPE):
        axes = tuple(index for index in range(8) if index != coordinate)
        assert tensor.sum(axis=axes) == pytest.approx([1 / bins] * bins, abs=2e-10)

    with pytest.raises(ValueError, match="exceeds fixed V8 support"):
        fidelity_v8.fit_behavior_profile_v8(
            calibration_fixture(tail_count=257), customer_ids_sha256="c" * 64
        )


def test_v8_preregistration_binds_lineage_and_owned_implementation_files() -> None:
    payload = fidelity_v8._preregistration_payload()
    root = Path(__file__).resolve().parents[1]
    immutable = {
        "results/entity_fidelity_v7_preregistration.json": fidelity_v8.PARENT_V7_PREREGISTRATION_SHA256,
        "results/entity_fidelity_v7_protocol.json": fidelity_v8.PARENT_V7_PROTOCOL_SHA256,
        "results/entity_fidelity_v7_development_attempt.json": fidelity_v8.PARENT_V7_ATTEMPT_SHA256,
        "results/entity_fidelity_v7_exploratory.json": fidelity_v8.PARENT_V7_EXPLORATORY_SHA256,
        "sim/entity_fidelity_v7.py": fidelity_v8.PARENT_V7_ENTITY_SOURCE_SHA256,
        "sim/population_v7.py": fidelity_v8.PARENT_V7_POPULATION_SOURCE_SHA256,
        "tests/test_entity_fidelity_v7.py": fidelity_v8.PARENT_V7_ENTITY_TEST_SHA256,
        "tests/test_population_v7.py": fidelity_v8.PARENT_V7_POPULATION_TEST_SHA256,
    }

    assert payload["schema_version"] == 8
    assert payload["study_revision"] == "8.0"
    assert payload["uci502_transactions_accessed"] is False
    assert payload["futility_folds"] == list(range(5))
    assert payload["futility_bootstrap_repeats"] == 10_000
    assert payload["full_bootstrap_seed"] == 20260857
    assert {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in immutable
    } == immutable
    assert {
        "sim/population_v8.py",
        "sim/entity_fidelity_v8.py",
        "tests/test_population_v8.py",
        "tests/test_entity_fidelity_v8.py",
    } <= set(fidelity_v8._implementation_hashes())
    assert all("_v8_" in path.name for path in (
        fidelity_v8.DEFAULT_PREREGISTRATION,
        fidelity_v8.DEFAULT_PROTOCOL,
        fidelity_v8.DEFAULT_EXPLORATORY,
        fidelity_v8.DEFAULT_DEVELOPMENT_ATTEMPT,
        fidelity_v8.DEFAULT_SEAL_RECEIPT,
        fidelity_v8.DEFAULT_OUTPUT,
        fidelity_v8.DEFAULT_COMPLETION_SEAL,
    ))


def test_preregistered_all_fold_futility_stops_before_other_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, ...] | None]] = []
    frame = pd.DataFrame(
        {"customer_id": ["one"], "ts": [pd.Timestamp("2010-01-01")], "amount": [1.0]}
    )

    monkeypatch.setattr(
        fidelity_v8,
        "fit_behavior_profile_v8",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="profile"),
    )

    def crossfit(
        _frame: object, *, seed: int, fold_numbers: object = None
    ) -> list[dict[str, object]]:
        normalized = tuple(fold_numbers) if fold_numbers is not None else None
        calls.append((seed, normalized))
        return [{"fold": fold} for fold in fidelity_v8.FUTILITY_FOLDS]

    monkeypatch.setattr(fidelity_v8, "_crossfit_seed", crossfit)
    monkeypatch.setattr(
        fidelity_v8, "_seed_futility", lambda _folds: _futility(passed=False)
    )
    result = fidelity_v8._compute_calibration_exploratory(
        frame,
        {"workbook_sha256": "workbook"},
        {"protocol_sha256": "p" * 64, "generator": {"behavior_profile_sha256": "profile"}},
    )

    assert calls == [(fidelity_v8.FUTILITY_SEED, None)]
    assert result["futility"]["passed_to_full_gate"] is False
    assert result["seeds"] == {}
    assert result["metrics"] == {}
    assert result["gate_passed"] is False
    assert fidelity_v8._futility_evidence_valid(result, require_pass=False)


def test_seed_futility_uses_frozen_all_fold_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    folds = [
        {
            "fold": fold,
            "evaluation_real_customers": 2,
            "evaluation_synthetic_customers": 2,
            "d": 0.6,
        }
        for fold in fidelity_v8.FUTILITY_FOLDS
    ]
    observed: dict[str, object] = {}

    def bootstrap(seed_folds: object, **kwargs: object) -> tuple[list[float], float, float]:
        observed["seed_folds"] = seed_folds
        observed.update(kwargs)
        return [0.6, 0.64], 0.6, 0.64

    monkeypatch.setattr(fidelity_v8, "_crossfit_max_seed_bootstrap", bootstrap)
    result = fidelity_v8._seed_futility(folds)

    assert observed == {
        "seed_folds": {fidelity_v8.FUTILITY_SEED: folds},
        "repeats": 10_000,
        "seed": 20260897,
    }
    assert result["folds"] == list(range(5))
    assert result["fold_weighted_d"] == pytest.approx(0.6)
    assert result["bootstrap_d_q_025"] == pytest.approx(0.601)
    assert result["passed_to_full_gate"] is True


def test_passing_futility_reuses_all_five_folds_in_full_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"customer_id": ["one"], "ts": [pd.Timestamp("2010-01-01")], "amount": [1.0]}
    )
    calls: list[int] = []
    produced: dict[int, list[dict[str, object]]] = {}
    reused = False

    monkeypatch.setattr(
        fidelity_v8,
        "fit_behavior_profile_v8",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="profile"),
    )

    def crossfit(
        _frame: object, *, seed: int, fold_numbers: object = None
    ) -> list[dict[str, object]]:
        assert fold_numbers is None
        calls.append(seed)
        produced[seed] = [
            {
                "fold": fold,
                "evaluation_real_customers": 1,
                "evaluation_synthetic_customers": 1,
                "auc": 0.6,
                "d": 0.6,
            }
            for fold in fidelity_v8.FUTILITY_FOLDS
        ]
        return produced[seed]

    def bootstrap(seed_folds: object, **_kwargs: object) -> tuple[list[float], float, float]:
        nonlocal reused
        reused = seed_folds[fidelity_v8.FUTILITY_SEED] is produced[fidelity_v8.FUTILITY_SEED]
        return [0.6], 0.6, 0.6

    monkeypatch.setattr(fidelity_v8, "_crossfit_seed", crossfit)
    monkeypatch.setattr(fidelity_v8, "_seed_futility", lambda _folds: _futility(passed=True))
    monkeypatch.setattr(fidelity_v8, "_crossfit_max_seed_bootstrap", bootstrap)
    result = fidelity_v8._compute_calibration_exploratory(
        frame,
        {"workbook_sha256": "workbook"},
        {"protocol_sha256": "p" * 64, "generator": {"behavior_profile_sha256": "profile"}},
    )

    assert calls.count(fidelity_v8.FUTILITY_SEED) == 1
    assert set(calls) == set(fidelity_v8.EVALUATION_SEEDS)
    assert reused is True
    assert result["gate_passed"] is True


def test_futility_stop_artifact_passes_schema_and_exclusive_write(tmp_path: Path) -> None:
    result = {
        "schema_version": 8,
        "artifact": "entity_fidelity_v8_calibration_exploratory_failure",
        "protocol_sha256": "a" * 64,
        "calibration_only": True,
        "calibration_access_attempted": True,
        "uci352_diagnostic_transactions_materialized_or_used": False,
        "uci502_transactions_accessed": False,
        "uci352_prior_holdout_transactions_materialized_or_used": False,
        "threshold_changed": False,
        "reference": {
            "workbook_sha256": fidelity_v8.UCI352_WORKBOOK_SHA256,
            "partition": "calibration",
            "customer_ids_sha256": fidelity_v8.EXPECTED_PARTITION_HASHES["calibration"],
            "customers": fidelity_v8.UCI352_CALIBRATION_CUSTOMERS,
            "transactions": fidelity_v8.UCI352_CALIBRATION_TRANSACTIONS,
        },
        "futility": _futility(passed=False),
        "seeds": {},
        "metrics": {},
        "gate_passed": False,
        "status": "STOP_FAIL",
        "failure": {
            "error_type": "futility_threshold_exceeded",
            "message": "preregistered futility lower bound exceeded the frozen threshold",
        },
    }
    output = tmp_path / "futility-failure.json"

    encoded = fidelity_v8._write_artifact(output, result)

    assert json.loads(encoded) == result
    assert json.loads(output.read_bytes()) == result
    assert oct(output.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError):
        fidelity_v8._write_artifact(output, result)


def test_failed_v8_exploratory_blocks_final_before_any_data_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "entity_fidelity_v8_protocol.json"
    exploratory_path = tmp_path / "entity_fidelity_v8_exploratory.json"
    attempt_path = tmp_path / "entity_fidelity_v8_development_attempt.json"
    protocol = {"protocol_sha256": "a" * 64}
    attempt = {
        "schema_version": 8,
        "artifact": "entity_fidelity_v8_development_attempt",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "consumed",
        "calibration_access_reserved": True,
        "maximum_calibration_loader_calls": 1,
        "uci502_transactions_accessed": False,
    }
    monkeypatch.setattr(fidelity_v8, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v8, "DEFAULT_EXPLORATORY", exploratory_path)
    monkeypatch.setattr(fidelity_v8, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(fidelity_v8, "DEFAULT_OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(fidelity_v8, "DEFAULT_SEAL_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(fidelity_v8, "DEFAULT_COMPLETION_SEAL", tmp_path / "seal.json")
    monkeypatch.setattr(fidelity_v8, "load_frozen_v8_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        fidelity_v8,
        "_read_regular_file_bytes",
        lambda path: json.dumps(attempt).encode() if path == attempt_path else b"{}",
    )
    monkeypatch.setattr(
        fidelity_v8,
        "load_retail_partition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data opened")),
    )

    with pytest.raises(ValueError, match="exploratory gate is absent"):
        fidelity_v8.run_entity_fidelity_v8_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=(
                f"{fidelity_v8.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
            ),
        )
