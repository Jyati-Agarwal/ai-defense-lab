from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import sim.entity_fidelity_v7 as fidelity_v7


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
        "seed": fidelity_v7.FUTILITY_SEED,
        "fold": fidelity_v7.FUTILITY_FOLD,
        "real_customers": 12,
        "synthetic_customers": 12,
        "auc": 0.6,
        "d": 0.6,
        "bootstrap_repeats": fidelity_v7.FUTILITY_BOOTSTRAP_REPEATS,
        "bootstrap_rng_seed": fidelity_v7.FUTILITY_BOOTSTRAP_SEED,
        "bootstrap_quantile": fidelity_v7.FUTILITY_QUANTILE,
        "bootstrap_d_mean": 0.67,
        "bootstrap_d_q_025": q_025,
        "threshold_max_d_upper_95": fidelity_v7.GATE_MAX_D_UPPER_95,
        "passed_to_full_gate": passed,
    }


def test_v7_fit_persists_exact_repeat_pmf_tail_and_rejects_overflow() -> None:
    profile = fidelity_v7.fit_behavior_profile_v7(
        calibration_fixture(), customer_ids_sha256="c" * 64
    )

    assert len(profile.repeat_invoice_count_pmf) == 257
    assert profile.repeat_invoice_count_pmf[:2] == (0.0, 0.0)
    assert profile.repeat_invoice_count_pmf[2] == 62 / 63
    assert profile.repeat_invoice_count_pmf[256] == 1 / 63
    assert sum(profile.repeat_invoice_count_pmf) == pytest.approx(1.0)
    assert profile.strata[1].primitive_quantiles[0][0] == 2
    assert profile.strata[1].primitive_quantiles[0][-1] == 256

    with pytest.raises(ValueError, match="exceeds fixed V7 support"):
        fidelity_v7.fit_behavior_profile_v7(
            calibration_fixture(tail_count=257), customer_ids_sha256="c" * 64
        )


def test_v7_preregistration_binds_lineage_and_owned_implementation_files() -> None:
    payload = fidelity_v7._preregistration_payload()
    root = Path(__file__).resolve().parents[1]
    immutable = {
        "results/entity_fidelity_v6_freeze_failure.json": fidelity_v7.PARENT_V6_FAILURE_SHA256,
        "results/entity_fidelity_v6_development_attempt.json": fidelity_v7.PARENT_V6_ATTEMPT_SHA256,
        "results/entity_fidelity_v61_preregistration.json": fidelity_v7.PARENT_V61_PREREGISTRATION_SHA256,
        "results/entity_fidelity_v61_protocol.json": fidelity_v7.PARENT_V61_PROTOCOL_SHA256,
        "results/entity_fidelity_v61_development_attempt.json": fidelity_v7.PARENT_V61_ATTEMPT_SHA256,
        "results/entity_fidelity_v61_exploratory.json": fidelity_v7.PARENT_V61_EXPLORATORY_SHA256,
    }

    assert payload["schema_version"] == 7
    assert payload["study_revision"] == "7.0"
    assert payload["uci502_transactions_accessed"] is False
    assert payload["futility_bootstrap_repeats"] == 10_000
    assert {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in immutable
    } == immutable
    assert {
        "sim/population_v7.py",
        "sim/entity_fidelity_v7.py",
        "tests/test_population_v7.py",
        "tests/test_entity_fidelity_v7.py",
    } <= set(fidelity_v7._implementation_hashes())
    assert all("_v7_" in path.name for path in (
        fidelity_v7.DEFAULT_PREREGISTRATION,
        fidelity_v7.DEFAULT_PROTOCOL,
        fidelity_v7.DEFAULT_EXPLORATORY,
        fidelity_v7.DEFAULT_DEVELOPMENT_ATTEMPT,
        fidelity_v7.DEFAULT_SEAL_RECEIPT,
        fidelity_v7.DEFAULT_OUTPUT,
        fidelity_v7.DEFAULT_COMPLETION_SEAL,
    ))


def test_preregistered_futility_stops_before_the_remaining_fourteen_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, ...] | None]] = []
    frame = pd.DataFrame(
        {"customer_id": ["one"], "ts": [pd.Timestamp("2010-01-01")], "amount": [1.0]}
    )

    monkeypatch.setattr(
        fidelity_v7,
        "fit_behavior_profile_v7",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="profile"),
    )

    def crossfit(
        _frame: object, *, seed: int, fold_numbers: object = None
    ) -> list[dict[str, object]]:
        normalized = tuple(fold_numbers) if fold_numbers is not None else None
        calls.append((seed, normalized))
        return [{"fold": fidelity_v7.FUTILITY_FOLD}]

    monkeypatch.setattr(fidelity_v7, "_crossfit_seed", crossfit)
    monkeypatch.setattr(
        fidelity_v7, "_representative_futility", lambda _fold: _futility(passed=False)
    )
    result = fidelity_v7._compute_calibration_exploratory(
        frame,
        {"workbook_sha256": "workbook"},
        {"protocol_sha256": "p" * 64, "generator": {"behavior_profile_sha256": "profile"}},
    )

    assert calls == [(fidelity_v7.FUTILITY_SEED, (fidelity_v7.FUTILITY_FOLD,))]
    assert result["futility"]["passed_to_full_gate"] is False
    assert result["seeds"] == {}
    assert result["metrics"] == {}
    assert result["gate_passed"] is False
    assert fidelity_v7._futility_evidence_valid(result, require_pass=False)


def test_futility_stop_artifact_passes_schema_and_exclusive_write(tmp_path: Path) -> None:
    result = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_calibration_exploratory_failure",
        "protocol_sha256": "a" * 64,
        "calibration_only": True,
        "calibration_access_attempted": True,
        "uci352_diagnostic_transactions_materialized_or_used": False,
        "uci502_transactions_accessed": False,
        "uci352_prior_holdout_transactions_materialized_or_used": False,
        "threshold_changed": False,
        "reference": {
            "workbook_sha256": fidelity_v7.UCI352_WORKBOOK_SHA256,
            "partition": "calibration",
            "customer_ids_sha256": fidelity_v7.EXPECTED_PARTITION_HASHES["calibration"],
            "customers": fidelity_v7.UCI352_CALIBRATION_CUSTOMERS,
            "transactions": fidelity_v7.UCI352_CALIBRATION_TRANSACTIONS,
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

    encoded = fidelity_v7._write_artifact(output, result)

    assert json.loads(encoded) == result
    assert json.loads(output.read_bytes()) == result
    assert oct(output.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError):
        fidelity_v7._write_artifact(output, result)


def test_failed_v7_exploratory_blocks_final_before_any_data_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "entity_fidelity_v7_protocol.json"
    exploratory_path = tmp_path / "entity_fidelity_v7_exploratory.json"
    attempt_path = tmp_path / "entity_fidelity_v7_development_attempt.json"
    protocol = {"protocol_sha256": "a" * 64}
    attempt = {
        "schema_version": 7,
        "artifact": "entity_fidelity_v7_development_attempt",
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "consumed",
        "calibration_access_reserved": True,
        "maximum_calibration_loader_calls": 1,
        "uci502_transactions_accessed": False,
    }
    monkeypatch.setattr(fidelity_v7, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v7, "DEFAULT_EXPLORATORY", exploratory_path)
    monkeypatch.setattr(fidelity_v7, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(fidelity_v7, "DEFAULT_OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(fidelity_v7, "DEFAULT_SEAL_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(fidelity_v7, "DEFAULT_COMPLETION_SEAL", tmp_path / "seal.json")
    monkeypatch.setattr(fidelity_v7, "load_frozen_v7_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        fidelity_v7,
        "_read_regular_file_bytes",
        lambda path: json.dumps(attempt).encode() if path == attempt_path else b"{}",
    )
    monkeypatch.setattr(
        fidelity_v7,
        "load_retail_partition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data opened")),
    )

    with pytest.raises(ValueError, match="exploratory gate is absent"):
        fidelity_v7.run_entity_fidelity_v7_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=(
                f"{fidelity_v7.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
            ),
        )
