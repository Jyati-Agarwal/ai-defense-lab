from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sim.entity_fidelity_v9 as fidelity_v9


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
        "seed": fidelity_v9.FUTILITY_SEED,
        "folds": list(fidelity_v9.FUTILITY_FOLDS),
        "real_customers": 12,
        "synthetic_customers": 12,
        "fold_weighted_d": 0.6,
        "bootstrap_repeats": fidelity_v9.FUTILITY_BOOTSTRAP_REPEATS,
        "bootstrap_rng_seed": fidelity_v9.FUTILITY_BOOTSTRAP_SEED,
        "bootstrap_quantile": fidelity_v9.FUTILITY_QUANTILE,
        "bootstrap_d_mean": 0.67,
        "bootstrap_d_q_025": q_025,
        "threshold_max_d_upper_95": fidelity_v9.GATE_MAX_D_UPPER_95,
        "passed_to_full_gate": passed,
    }


def test_v9_fit_persists_exact_repeat_pmf_tail_and_rejects_overflow() -> None:
    profile, primitive_rows = fidelity_v9._fit_behavior_profile_and_rows(
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
        fidelity_v9.REPEAT_CHECKERBOARD_SHAPE
    )
    source = np.asarray(
        fidelity_v9.fractional_repeat_masses_v8(
            primitive_rows["multiday_repeat"]
        )
    ).reshape(fidelity_v9.REPEAT_CHECKERBOARD_SHAPE)
    assert tensor.sum(axis=0) == pytest.approx(source.sum(axis=0), abs=1e-12)
    for coordinate in range(1, 8):
        axes = tuple(index for index in range(8) if index not in (0, coordinate))
        assert tensor.sum(axis=axes) == pytest.approx(
            source.sum(axis=axes), abs=1e-12
        )
    support = np.broadcast_to(source.sum(axis=0) > 0, source.shape).copy()
    for coordinate in range(1, 8):
        axes = tuple(index for index in range(8) if index not in (0, coordinate))
        target = source.sum(axis=axes)
        shape = [1] * 8
        shape[0] = 32
        shape[coordinate] = 3
        support &= np.broadcast_to((target > 0).reshape(shape), source.shape)
    assert np.all(tensor[~support] == 0)
    source_positive = source > 0
    tensor_positive = tensor > 0
    assert -np.sum(tensor[tensor_positive] * np.log(tensor[tensor_positive])) >= (
        -np.sum(source[source_positive] * np.log(source[source_positive])) - 1e-10
    )
    for coordinate, bins in enumerate(fidelity_v9.REPEAT_CHECKERBOARD_SHAPE):
        axes = tuple(index for index in range(8) if index != coordinate)
        assert tensor.sum(axis=axes) == pytest.approx([1 / bins] * bins, abs=2e-10)

    with pytest.raises(ValueError, match="exceeds fixed V9 support"):
        fidelity_v9.fit_behavior_profile_v9(
            calibration_fixture(tail_count=257), customer_ids_sha256="c" * 64
        )


def test_v9_ipf_removes_unidentified_interaction_and_fails_closed() -> None:
    shape = fidelity_v9.REPEAT_CHECKERBOARD_SHAPE
    uniform = np.full(shape, 1 / np.prod(shape))
    count = np.linspace(-1.0, 1.0, shape[0])
    thirds = np.asarray((-1.0, 0.0, 1.0))
    pure_three_way = uniform * (
        1
        + 0.5
        * count[:, None, None, None, None, None, None, None]
        * thirds[None, :, None, None, None, None, None, None]
        * thirds[None, None, :, None, None, None, None, None]
    )

    first = fidelity_v9.maximum_entropy_projection_v9(pure_three_way)
    second = fidelity_v9.maximum_entropy_projection_v9(pure_three_way)

    assert np.array_equal(first, second)
    assert first.flags.c_contiguous
    assert first == pytest.approx(uniform, abs=1e-15)
    arbitrary_zero = uniform.copy()
    arbitrary_zero[(0,) * 8] = 0
    assert fidelity_v9.maximum_entropy_projection_v9(arbitrary_zero)[(0,) * 8] > 0
    invalid = pure_three_way.copy()
    invalid.flat[0] = np.nan
    with pytest.raises(ValueError, match="IPF inputs"):
        fidelity_v9.maximum_entropy_projection_v9(invalid)

    two_pairs = uniform * (
        1
        + 0.35
        * count[:, None, None, None, None, None, None, None]
        * thirds[None, :, None, None, None, None, None, None]
        + 0.35
        * count[:, None, None, None, None, None, None, None]
        * thirds[None, None, :, None, None, None, None, None]
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        fidelity_v9.maximum_entropy_projection_v9(two_pairs, max_sweeps=1)


def test_v9_preregistration_binds_lineage_and_owned_implementation_files() -> None:
    payload = fidelity_v9._preregistration_payload()
    root = Path(__file__).resolve().parents[1]
    immutable = {
        "results/entity_fidelity_v8_preregistration.json": fidelity_v9.PARENT_V8_PREREGISTRATION_SHA256,
        "results/entity_fidelity_v8_protocol.json": fidelity_v9.PARENT_V8_PROTOCOL_SHA256,
        "results/entity_fidelity_v8_development_attempt.json": fidelity_v9.PARENT_V8_ATTEMPT_SHA256,
        "results/entity_fidelity_v8_exploratory.json": fidelity_v9.PARENT_V8_EXPLORATORY_SHA256,
        "sim/entity_fidelity_v8.py": fidelity_v9.PARENT_V8_ENTITY_SOURCE_SHA256,
        "sim/population_v8.py": fidelity_v9.PARENT_V8_POPULATION_SOURCE_SHA256,
        "tests/test_entity_fidelity_v8.py": fidelity_v9.PARENT_V8_ENTITY_TEST_SHA256,
        "tests/test_population_v8.py": fidelity_v9.PARENT_V8_POPULATION_TEST_SHA256,
    }

    assert payload["schema_version"] == 9
    assert payload["study_revision"] == "9.0"
    assert payload["model_family_terminal_if_failed"] is True
    assert payload["uci502_transactions_accessed"] is False
    assert payload["futility_folds"] == list(range(5))
    assert payload["futility_bootstrap_repeats"] == 10_000
    assert payload["full_bootstrap_seed"] == 20260857
    assert {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in immutable
    } == immutable
    assert {
        "sim/population_v9.py",
        "sim/entity_fidelity_v9.py",
        "tests/test_population_v9.py",
        "tests/test_entity_fidelity_v9.py",
    } <= set(fidelity_v9._implementation_hashes())
    assert all("_v9_" in path.name for path in (
        fidelity_v9.DEFAULT_PREREGISTRATION,
        fidelity_v9.DEFAULT_PROTOCOL,
        fidelity_v9.DEFAULT_EXPLORATORY,
        fidelity_v9.DEFAULT_DEVELOPMENT_ATTEMPT,
        fidelity_v9.DEFAULT_SEAL_RECEIPT,
        fidelity_v9.DEFAULT_OUTPUT,
        fidelity_v9.DEFAULT_COMPLETION_SEAL,
    ))


def test_preregistered_all_fold_futility_stops_before_other_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, ...] | None]] = []
    frame = pd.DataFrame(
        {"customer_id": ["one"], "ts": [pd.Timestamp("2010-01-01")], "amount": [1.0]}
    )

    monkeypatch.setattr(
        fidelity_v9,
        "fit_behavior_profile_v9",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="profile"),
    )

    def crossfit(
        _frame: object, *, seed: int, fold_numbers: object = None
    ) -> list[dict[str, object]]:
        normalized = tuple(fold_numbers) if fold_numbers is not None else None
        calls.append((seed, normalized))
        return [{"fold": fold} for fold in fidelity_v9.FUTILITY_FOLDS]

    monkeypatch.setattr(fidelity_v9, "_crossfit_seed", crossfit)
    monkeypatch.setattr(
        fidelity_v9, "_seed_futility", lambda _folds: _futility(passed=False)
    )
    result = fidelity_v9._compute_calibration_exploratory(
        frame,
        {"workbook_sha256": "workbook"},
        {"protocol_sha256": "p" * 64, "generator": {"behavior_profile_sha256": "profile"}},
    )

    assert calls == [(fidelity_v9.FUTILITY_SEED, None)]
    assert result["futility"]["passed_to_full_gate"] is False
    assert result["seeds"] == {}
    assert result["metrics"] == {}
    assert result["gate_passed"] is False
    assert fidelity_v9._futility_evidence_valid(result, require_pass=False)


def test_seed_futility_uses_frozen_all_fold_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    folds = [
        {
            "fold": fold,
            "evaluation_real_customers": 2,
            "evaluation_synthetic_customers": 2,
            "d": 0.6,
        }
        for fold in fidelity_v9.FUTILITY_FOLDS
    ]
    observed: dict[str, object] = {}

    def bootstrap(seed_folds: object, **kwargs: object) -> tuple[list[float], float, float]:
        observed["seed_folds"] = seed_folds
        observed.update(kwargs)
        return [0.6, 0.64], 0.6, 0.64

    monkeypatch.setattr(fidelity_v9, "_crossfit_max_seed_bootstrap", bootstrap)
    result = fidelity_v9._seed_futility(folds)

    assert observed == {
        "seed_folds": {fidelity_v9.FUTILITY_SEED: folds},
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
        fidelity_v9,
        "fit_behavior_profile_v9",
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
            for fold in fidelity_v9.FUTILITY_FOLDS
        ]
        return produced[seed]

    def bootstrap(seed_folds: object, **_kwargs: object) -> tuple[list[float], float, float]:
        nonlocal reused
        reused = seed_folds[fidelity_v9.FUTILITY_SEED] is produced[fidelity_v9.FUTILITY_SEED]
        return [0.6], 0.6, 0.6

    monkeypatch.setattr(fidelity_v9, "_crossfit_seed", crossfit)
    monkeypatch.setattr(fidelity_v9, "_seed_futility", lambda _folds: _futility(passed=True))
    monkeypatch.setattr(fidelity_v9, "_crossfit_max_seed_bootstrap", bootstrap)
    result = fidelity_v9._compute_calibration_exploratory(
        frame,
        {"workbook_sha256": "workbook"},
        {"protocol_sha256": "p" * 64, "generator": {"behavior_profile_sha256": "profile"}},
    )

    assert calls.count(fidelity_v9.FUTILITY_SEED) == 1
    assert set(calls) == set(fidelity_v9.EVALUATION_SEEDS)
    assert reused is True
    assert result["gate_passed"] is True


def test_futility_stop_artifact_passes_schema_and_exclusive_write(tmp_path: Path) -> None:
    result = {
        "schema_version": 9,
        "artifact": "entity_fidelity_v9_calibration_exploratory_failure",
        "protocol_sha256": "a" * 64,
        "calibration_only": True,
        "calibration_access_attempted": True,
        "uci352_diagnostic_transactions_materialized_or_used": False,
        "uci502_transactions_accessed": False,
        "uci352_prior_holdout_transactions_materialized_or_used": False,
        "threshold_changed": False,
        "reference": {
            "workbook_sha256": fidelity_v9.UCI352_WORKBOOK_SHA256,
            "partition": "calibration",
            "customer_ids_sha256": fidelity_v9.EXPECTED_PARTITION_HASHES["calibration"],
            "customers": fidelity_v9.UCI352_CALIBRATION_CUSTOMERS,
            "transactions": fidelity_v9.UCI352_CALIBRATION_TRANSACTIONS,
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

    encoded = fidelity_v9._write_artifact(output, result)

    assert json.loads(encoded) == result
    assert json.loads(output.read_bytes()) == result
    assert oct(output.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError):
        fidelity_v9._write_artifact(output, result)


def test_v9_existing_attempt_stops_before_any_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.xlsx"
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "exploratory.json"
    attempt = tmp_path / "attempt.json"
    attempt.write_bytes(b"reserved")
    monkeypatch.setattr(fidelity_v9, "DEFAULT_REFERENCE", reference)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_PROTOCOL", protocol)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_EXPLORATORY", output)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(
        fidelity_v9,
        "load_frozen_v9_protocol",
        lambda *_args: (_ for _ in ()).throw(AssertionError("protocol opened")),
    )
    monkeypatch.setattr(
        fidelity_v9,
        "load_retail_partition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data opened")),
    )

    with pytest.raises(FileExistsError, match="already consumed"):
        fidelity_v9.run_calibration_exploratory(
            reference, output, protocol_path=protocol
        )


def test_v9_receipt_is_persisted_before_uci502_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sim.entity_fidelity_v4 as fidelity_v4

    protocol_path = tmp_path / "protocol.json"
    exploratory_path = tmp_path / "exploratory.json"
    attempt_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    protocol = {
        "protocol_sha256": "a" * 64,
        "generator": {"behavior_profile_sha256": "profile"},
    }
    state: dict[str, object] = {}
    calibration = pd.DataFrame(
        {
            "customer_id": ["one"],
            "ts": [pd.Timestamp("2010-01-01")],
            "amount": [1.0],
        }
    )
    monkeypatch.setattr(fidelity_v9, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_EXPLORATORY", exploratory_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(fidelity_v9, "DEFAULT_SEAL_RECEIPT", receipt_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_COMPLETION_SEAL", tmp_path / "seal.json")
    monkeypatch.setattr(fidelity_v9, "load_frozen_v9_protocol", lambda _path: protocol)
    monkeypatch.setattr(fidelity_v9, "_development_attempt_valid", lambda *_args: True)
    monkeypatch.setattr(
        fidelity_v9,
        "_recompute_and_match_exploratory",
        lambda *_args: (calibration, {}, {"gate_passed": True}, "b" * 64),
    )
    monkeypatch.setattr(
        fidelity_v9,
        "fit_behavior_profile_v9",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="profile"),
    )
    monkeypatch.setattr(
        fidelity_v9,
        "_ids_hash",
        lambda _ids: fidelity_v9.EXPECTED_PARTITION_HASHES["calibration"],
    )
    monkeypatch.setattr(fidelity_v9, "_final_receipt_valid", lambda *_args, **_kwargs: True)

    def read(path: Path) -> bytes:
        if path == attempt_path:
            return b"{}"
        assert path == receipt_path
        return state["receipt_bytes"]  # type: ignore[return-value]

    def write(_path: Path, value: object) -> bytes:
        encoded = fidelity_v9._pretty_json_bytes(value)  # type: ignore[arg-type]
        state["receipt_artifact"] = value["artifact"]  # type: ignore[index]
        state["receipt_bytes"] = encoded
        return encoded

    def uci502_loader(*_args: object, **_kwargs: object) -> object:
        assert state["receipt_artifact"] == "entity_fidelity_v9_one_shot_final_receipt"
        raise RuntimeError("uci502 loader reached after receipt")

    monkeypatch.setattr(fidelity_v9, "_read_regular_file_bytes", read)
    monkeypatch.setattr(fidelity_v9, "_write_artifact", write)
    monkeypatch.setattr(fidelity_v4, "load_retail_ii_readiness", uci502_loader)

    with pytest.raises(RuntimeError, match="after receipt"):
        fidelity_v9.run_entity_fidelity_v9_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=(
                f"{fidelity_v9.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
            ),
        )


def test_failed_v9_exploratory_blocks_final_before_any_data_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "entity_fidelity_v9_protocol.json"
    exploratory_path = tmp_path / "entity_fidelity_v9_exploratory.json"
    attempt_path = tmp_path / "entity_fidelity_v9_development_attempt.json"
    protocol = {"protocol_sha256": "a" * 64}
    attempt = {
        "schema_version": 9,
        "artifact": "entity_fidelity_v9_development_attempt",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "consumed",
        "calibration_access_reserved": True,
        "maximum_calibration_loader_calls": 1,
        "uci502_transactions_accessed": False,
    }
    monkeypatch.setattr(fidelity_v9, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_EXPLORATORY", exploratory_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(fidelity_v9, "DEFAULT_OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(fidelity_v9, "DEFAULT_SEAL_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(fidelity_v9, "DEFAULT_COMPLETION_SEAL", tmp_path / "seal.json")
    monkeypatch.setattr(fidelity_v9, "load_frozen_v9_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        fidelity_v9,
        "_read_regular_file_bytes",
        lambda path: json.dumps(attempt).encode() if path == attempt_path else b"{}",
    )
    monkeypatch.setattr(
        fidelity_v9,
        "load_retail_partition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data opened")),
    )

    with pytest.raises(ValueError, match="exploratory gate is absent"):
        fidelity_v9.run_entity_fidelity_v9_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=(
                f"{fidelity_v9.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
            ),
        )
