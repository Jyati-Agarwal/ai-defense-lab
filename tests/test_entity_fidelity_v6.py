from __future__ import annotations

import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

import sim.entity_fidelity_v4 as fidelity_v4
import sim.entity_fidelity_v6 as fidelity_v6
from sim.entity_fidelity import _ids_hash, _transactions_hash


def calibration_fixture(
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
            [
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
            ]
        )
    for index in range(multiday):
        day = base + pd.Timedelta(days=index)
        rows.extend(
            [
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
            ]
        )
    return pd.DataFrame(rows).sort_values(["customer_id", "ts"], kind="stable")


def _profile() -> fidelity_v6.BehaviorProfileV6:
    frame = calibration_fixture()
    return fidelity_v6.fit_behavior_profile_v6(
        frame, customer_ids_sha256=_ids_hash(frame["customer_id"].unique())
    )


def _exploratory_result(protocol_sha256: str, *, passed: bool) -> dict[str, object]:
    return {
        "schema_version": 6,
        "artifact": "entity_fidelity_v6_calibration_exploratory",
        "protocol_sha256": protocol_sha256,
        "calibration_only": True,
        "uci352_diagnostic_transactions_materialized_or_used": False,
        "uci502_transactions_accessed": False,
        "uci352_prior_holdout_transactions_materialized_or_used": False,
        "reference": {},
        "seeds": {},
        "metrics": {},
        "gate_passed": passed,
    }


def _attempt(protocol_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 6,
        "artifact": "entity_fidelity_v6_development_attempt",
        "protocol_sha256": protocol_sha256,
        "status": "consumed",
        "calibration_access_reserved": True,
        "maximum_calibration_loader_calls": 1,
        "uci502_transactions_accessed": False,
    }


def test_v6_fit_has_exact_mixed_grid_time_coordinates_and_raw_resultant() -> None:
    frame = calibration_fixture()
    profile, rows = fidelity_v6._fit_behavior_profile_and_rows(
        frame, customer_ids_sha256="c" * 64
    )

    assert {name: len(values) for name, values in rows.items()} == {
        "singleton": 32,
        "multiday_repeat": 32,
    }
    assert [len(item.cell_masses) for item in profile.strata] == [216, 6561]
    assert [len(item.primitive_quantiles) for item in profile.strata] == [2, 8]
    assert profile.repeat_customer_count == 63
    assert math.isclose(profile.same_day_repeat_probability, 31 / 63)
    assert isinstance(profile.observation_days, int)
    assert math.isclose(
        profile.global_log_amount_center,
        float(frame["amount"].map(math.log).mean()),
        abs_tol=1e-12,
    )
    repeat = rows["multiday_repeat"]
    assert (repeat[:, 1] > 0).all() and (repeat[:, 1] < 1).all()
    assert (repeat[:, 2] > 0).all() and (repeat[:, 2] < 1).all()
    assert (repeat[:, 7] >= 0).all() and (repeat[:, 7] <= 1).all()


@pytest.mark.parametrize(
    ("singletons", "multiday", "stratum"),
    ((31, 32, "singleton"), (32, 31, "multiday_repeat")),
)
def test_v6_checkerboard_minima_are_fail_closed(
    singletons: int, multiday: int, stratum: str
) -> None:
    with pytest.raises(ValueError, match=rf"{stratum} has 31 customers"):
        fidelity_v6._fit_behavior_profile_and_rows(
            calibration_fixture(singletons=singletons, multiday=multiday),
            customer_ids_sha256="c" * 64,
        )


def test_discrete_invoice_knots_preserve_rare_tail_extrema() -> None:
    frame = calibration_fixture()
    frame = frame.loc[frame["customer_id"] != "multi-31"].copy()
    base = pd.Timestamp("2010-02-01")
    tail = pd.DataFrame(
        [
            {
                "customer_id": "multi-31",
                "ts": base + pd.Timedelta(minutes=30 * index),
                "amount": 61.0 + index / 100,
            }
            for index in range(100)
        ]
    )
    frame = pd.concat((frame, tail), ignore_index=True).sort_values(
        ["customer_id", "ts"], kind="stable"
    )
    profile = fidelity_v6.fit_behavior_profile_v6(
        frame, customer_ids_sha256="c" * 64
    )
    invoice_knots = profile.strata[1].primitive_quantiles[0]

    assert invoice_knots[0] == 2
    assert invoice_knots[-1] == 100
    assert all(float(value).is_integer() for value in invoice_knots)


def test_v6_novelty_requirement_and_aggregate_privacy() -> None:
    frame = calibration_fixture()
    profile, rows = fidelity_v6._fit_behavior_profile_and_rows(
        frame, customer_ids_sha256="c" * 64
    )
    result = fidelity_v6.novelty_check(
        profile, rows, draws=2_000, seed=71, batch_size=250
    )

    assert fidelity_v6.NOVELTY_DRAWS == 1_000_000
    assert result["exact_full_continuous_primitive_tuple_matches"] == 0
    assert result["passed"] is True
    fidelity_v6._assert_aggregate_only({"behavior_profile": profile.as_dict()})
    for key in ("customer_id", "raw_rows", "predictions", "feature_rows"):
        with pytest.raises(ValueError, match=r"not (?:allowed|an allowlisted)"):
            fidelity_v6._assert_aggregate_only(
                {key: [{"customer_id": "forbidden", "value": 1.0}]}
            )


def test_frozen_protocol_binds_v6_shapes_algorithms_and_unchanged_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = calibration_fixture()
    ids_hash = _ids_hash(frame["customer_id"].unique())
    transaction_hash = _transactions_hash(frame)
    workbook_hash = "d" * 64
    manifest_hash = "e" * 64
    monkeypatch.setattr(
        fidelity_v6,
        "load_retail_partition",
        lambda _path, partition: (
            frame,
            {
                "workbook_sha256": workbook_hash,
                "manifest_sha256": manifest_hash,
                "partition": partition,
            },
        ),
    )
    monkeypatch.setattr(
        fidelity_v6,
        "novelty_check",
        lambda *_args, **_kwargs: {
            "draws": 1_000_000,
            "exact_full_continuous_primitive_tuple_matches": 0,
            "passed": True,
        },
    )
    monkeypatch.setattr(fidelity_v6, "UCI352_WORKBOOK_SHA256", workbook_hash)
    monkeypatch.setattr(fidelity_v6, "UCI352_MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(
        fidelity_v6, "UCI352_CALIBRATION_CUSTOMERS", frame["customer_id"].nunique()
    )
    monkeypatch.setattr(
        fidelity_v6, "UCI352_CALIBRATION_TRANSACTIONS", len(frame)
    )
    monkeypatch.setattr(
        fidelity_v6, "UCI352_CALIBRATION_TRANSACTIONS_SHA256", transaction_hash
    )
    monkeypatch.setitem(
        fidelity_v6.EXPECTED_PARTITION_HASHES, "calibration", ids_hash
    )
    path = tmp_path / "protocol.json"
    protocol = fidelity_v6.freeze_v6_protocol(tmp_path / "never-opened.xlsx", path)

    assert protocol["generator"]["strata"]["singleton"]["cell_shape"] == [3, 3, 24]
    assert protocol["generator"]["strata"]["multiday_repeat"]["cell_shape"] == [3] * 8
    assert protocol["generator"]["time_coordinates"] == fidelity_v6.TIME_COORDINATES
    assert protocol["generator"]["hour_allocation"] == fidelity_v6.HOUR_ALLOCATION
    assert protocol["generator"]["canonical_allocator"] == fidelity_v6.CANONICAL_ALLOCATION
    assert protocol["generator"]["amount_rendering"] == fidelity_v6.AMOUNT_RENDERING
    assert tuple(protocol["evaluation"]["features"]) == fidelity_v6.FEATURES
    assert tuple(protocol["evaluation"]["seeds"]) == fidelity_v6.EVALUATION_SEEDS
    assert protocol["evaluation"]["exploratory_crossfit_folds"] == 5
    assert protocol["evaluation"]["bootstrap_repeats"] == 10_000
    assert protocol["evaluation"]["threshold_max_d_upper_95"] == 0.65
    assert "data/market_prices.csv" in protocol["implementation_sha256"]
    assert "sim/mandate_twin.py" in protocol["implementation_sha256"]
    assert "cryptography" in protocol["dependency_versions"]
    assert protocol["development_gate"][
        "attempt_reserved_before_calibration_access"
    ] is True
    assert protocol["one_shot_final"]["completion_seal_binds_exact_final_bytes"] is True
    assert protocol["one_shot_final"][
        "external_signed_or_worm_custody_is_deployment_responsibility"
    ] is True
    assert protocol["one_shot_final"][
        "external_signed_or_worm_custody_is_release_prerequisite"
    ] is True
    assert fidelity_v6.load_frozen_v6_protocol(path) == protocol
    assert oct(path.stat().st_mode & 0o777) == "0o444"

    poisoned_protocol = json.loads(json.dumps(protocol))
    poisoned_protocol["generator"]["novelty"]["private_scalar"] = (
        "customer-secret"
    )
    poisoned_protocol["protocol_sha256"] = fidelity_v6._protocol_hash(
        poisoned_protocol
    )
    path.chmod(0o644)
    path.write_bytes(fidelity_v6._pretty_json_bytes(poisoned_protocol))
    with pytest.raises(ValueError, match="novelty gate"):
        fidelity_v6.load_frozen_v6_protocol(path)

    coerced_profile_protocol = json.loads(json.dumps(protocol))
    source = coerced_profile_protocol["generator"]["behavior_profile"]["source"]
    source["customers"] = str(source["customers"])
    coerced_profile_protocol["protocol_sha256"] = fidelity_v6._protocol_hash(
        coerced_profile_protocol
    )
    path.write_bytes(fidelity_v6._pretty_json_bytes(coerced_profile_protocol))
    with pytest.raises(ValueError, match="generator contract"):
        fidelity_v6.load_frozen_v6_protocol(path)


def test_development_gate_is_canonical_single_use_without_uci502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "calibration.xlsx"
    protocol_path = tmp_path / "protocol.json"
    output = tmp_path / "exploratory.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(fidelity_v6, "DEFAULT_REFERENCE", reference)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_EXPLORATORY", output)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(
        fidelity_v6,
        "load_frozen_v6_protocol",
        lambda _path: {"protocol_sha256": "a" * 64},
    )
    calls = {"calibration": 0}

    def calibration_loader(_path: Path, _partition: str) -> tuple[object, object]:
        calls["calibration"] += 1
        return object(), object()

    monkeypatch.setattr(fidelity_v6, "load_retail_partition", calibration_loader)
    monkeypatch.setattr(fidelity_v6, "_exploratory_evidence_valid", lambda _value: True)
    monkeypatch.setattr(
        fidelity_v6,
        "_compute_calibration_exploratory",
        lambda *_args: _exploratory_result("a" * 64, passed=False),
    )

    result = fidelity_v6.run_calibration_exploratory(
        reference, output, protocol_path=protocol_path
    )
    assert result["gate_passed"] is False
    assert result["status"] == "STOP_FAIL"
    assert result["threshold_changed"] is False
    assert calls == {"calibration": 1}
    assert attempt.exists()
    assert json.loads(attempt.read_text(encoding="utf-8")) == _attempt("a" * 64)
    assert oct(attempt.stat().st_mode & 0o777) == "0o444"
    assert oct(output.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError, match="already consumed"):
        fidelity_v6.run_calibration_exploratory(
            reference, output, protocol_path=protocol_path
        )
    assert calls == {"calibration": 1}
    assert not hasattr(fidelity_v6, "load_retail_ii_final")


def test_development_attempt_reservation_allows_one_concurrent_calibration_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "calibration.xlsx"
    protocol_path = tmp_path / "protocol.json"
    output = tmp_path / "exploratory.json"
    attempt = tmp_path / "attempt.json"
    for name, value in (
        ("DEFAULT_REFERENCE", reference),
        ("DEFAULT_PROTOCOL", protocol_path),
        ("DEFAULT_EXPLORATORY", output),
        ("DEFAULT_DEVELOPMENT_ATTEMPT", attempt),
    ):
        monkeypatch.setattr(fidelity_v6, name, value)
    rendezvous = threading.Barrier(2)

    def load_protocol(_path: Path) -> dict[str, str]:
        rendezvous.wait(timeout=5)
        return {"protocol_sha256": "c" * 64}

    calls = 0
    calls_lock = threading.Lock()

    def calibration_loader(_path: Path, _partition: str) -> tuple[object, object]:
        nonlocal calls
        with calls_lock:
            calls += 1
        return object(), object()

    monkeypatch.setattr(fidelity_v6, "load_frozen_v6_protocol", load_protocol)
    monkeypatch.setattr(fidelity_v6, "load_retail_partition", calibration_loader)
    monkeypatch.setattr(fidelity_v6, "_exploratory_evidence_valid", lambda _value: True)
    monkeypatch.setattr(
        fidelity_v6,
        "_compute_calibration_exploratory",
        lambda *_args: _exploratory_result("c" * 64, passed=True),
    )

    def invoke() -> str:
        try:
            fidelity_v6.run_calibration_exploratory(
                reference, output, protocol_path=protocol_path
            )
        except FileExistsError:
            return "blocked"
        return "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _index: invoke(), range(2)))

    assert outcomes == ["blocked", "completed"]
    assert calls == 1
    assert json.loads(attempt.read_text(encoding="utf-8")) == _attempt("c" * 64)


def test_keyboard_interrupt_burns_development_attempt_and_writes_sanitized_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "calibration.xlsx"
    protocol_path = tmp_path / "protocol.json"
    output = tmp_path / "exploratory.json"
    attempt = tmp_path / "attempt.json"
    for name, value in (
        ("DEFAULT_REFERENCE", reference),
        ("DEFAULT_PROTOCOL", protocol_path),
        ("DEFAULT_EXPLORATORY", output),
        ("DEFAULT_DEVELOPMENT_ATTEMPT", attempt),
    ):
        monkeypatch.setattr(fidelity_v6, name, value)
    monkeypatch.setattr(
        fidelity_v6,
        "load_frozen_v6_protocol",
        lambda _path: {"protocol_sha256": "d" * 64},
    )
    calls = 0

    def interrupted_loader(_path: Path, _partition: str) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt("private-calibration-value")

    monkeypatch.setattr(fidelity_v6, "load_retail_partition", interrupted_loader)
    with pytest.raises(KeyboardInterrupt, match="private-calibration-value"):
        fidelity_v6.run_calibration_exploratory(
            reference, output, protocol_path=protocol_path
        )

    failure = json.loads(output.read_text(encoding="utf-8"))
    assert set(failure) == fidelity_v6.FAILURE_EXPLORATORY_KEYS
    assert failure["failure"]["error_type"] == "interrupted"
    assert "private-calibration-value" not in json.dumps(failure)
    with pytest.raises(FileExistsError, match="already consumed"):
        fidelity_v6.run_calibration_exploratory(
            reference, output, protocol_path=protocol_path
        )
    assert calls == 1
    assert attempt.exists()


def test_artifact_writer_rejects_unknown_names_and_top_level_scalar_leaks(
    tmp_path: Path,
) -> None:
    passed = _exploratory_result("e" * 64, passed=True)
    failure = {
        **_exploratory_result("e" * 64, passed=False),
        "artifact": "entity_fidelity_v6_calibration_exploratory_failure",
        "status": "STOP_FAIL",
        "calibration_access_attempted": True,
        "threshold_changed": False,
        "failure": {"error_type": "test", "message": "sanitized"},
    }
    for index, artifact in enumerate((passed, failure)):
        leaked = {**artifact, "private_scalar": "customer-secret"}
        with pytest.raises(ValueError, match="top-level fields"):
            fidelity_v6._write_artifact(tmp_path / f"leaked-{index}.json", leaked)
    poisoned = {
        key: "customer-secret" for key in fidelity_v6.PASSED_EXPLORATORY_KEYS
    }
    poisoned.update(
        schema_version=6,
        artifact="entity_fidelity_v6_calibration_exploratory",
    )
    with pytest.raises(ValueError, match="values differ"):
        fidelity_v6._write_artifact(tmp_path / "poisoned.json", poisoned)
    invalid_attempt = _attempt("e" * 64)
    invalid_attempt["maximum_calibration_loader_calls"] = True
    with pytest.raises(ValueError, match="values differ"):
        fidelity_v6._write_artifact(
            tmp_path / "boolean-loader-limit.json", invalid_attempt
        )
    unknown = {"schema_version": 6, "artifact": "unregistered_v6_artifact"}
    with pytest.raises(ValueError, match="unknown v6 artifact"):
        fidelity_v6._write_artifact(tmp_path / "unknown.json", unknown)
    assert not any(tmp_path.glob("leaked-*.json"))
    assert not (tmp_path / "poisoned.json").exists()
    assert not (tmp_path / "boolean-loader-limit.json").exists()
    assert not (tmp_path / "unknown.json").exists()


def test_one_shot_receipt_precedes_uci502_and_crash_burns_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    exploratory_path = tmp_path / "exploratory.json"
    attempt_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    final_path = tmp_path / "final.json"
    completion_path = tmp_path / "completion.json"
    reference = tmp_path / "uci352.xlsx"
    final_reference = tmp_path / "uci502.xlsx"
    exploratory_path.write_text("{}\n", encoding="utf-8")
    profile = _profile()
    monkeypatch.setitem(
        fidelity_v6.EXPECTED_PARTITION_HASHES,
        "calibration",
        profile.source_customer_ids_sha256,
    )
    protocol = {
        "protocol_sha256": "a" * 64,
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    monkeypatch.setattr(fidelity_v6, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_EXPLORATORY", exploratory_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_SEAL_RECEIPT", receipt_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_OUTPUT", final_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_COMPLETION_SEAL", completion_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_REFERENCE", reference)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_FINAL_REFERENCE", final_reference)
    monkeypatch.setattr(fidelity_v6, "load_frozen_v6_protocol", lambda _path: protocol)
    fidelity_v6._write_artifact(attempt_path, _attempt(protocol["protocol_sha256"]))
    exploratory_sha256 = hashlib.sha256(exploratory_path.read_bytes()).hexdigest()
    recomputed = {"gate_passed": True}

    def verified_then_mutated(*_args: object) -> tuple[object, object, object, str]:
        exploratory_path.write_text('{"unchecked":"replacement"}\n', encoding="utf-8")
        return calibration_fixture(), {}, recomputed, exploratory_sha256

    monkeypatch.setattr(
        fidelity_v6,
        "_recompute_and_match_exploratory",
        verified_then_mutated,
    )
    monkeypatch.setattr(fidelity_v6, "fit_behavior_profile_v6", lambda *_args, **_kwargs: profile)
    observed = {"loader_calls": 0, "receipt_seen": False}

    def crashing_final_loader(*_args: object, **_kwargs: object) -> object:
        observed["loader_calls"] += 1
        observed["receipt_seen"] = receipt_path.exists()
        raise RuntimeError("synthetic UCI502 loader crash")

    monkeypatch.setattr(
        fidelity_v4, "load_retail_ii_readiness", crashing_final_loader
    )
    confirmation = f"{fidelity_v6.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
    with pytest.raises(RuntimeError, match="synthetic UCI502 loader crash"):
        fidelity_v6.run_entity_fidelity_v6_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=confirmation,
        )

    assert observed == {"loader_calls": 1, "receipt_seen": True}
    assert receipt_path.exists() and not final_path.exists()
    assert not completion_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "consumed"
    assert receipt["exploratory_artifact_sha256"] == exploratory_sha256
    assert receipt["exploratory_artifact_sha256"] != hashlib.sha256(
        exploratory_path.read_bytes()
    ).hexdigest()
    assert receipt["receipt_before_final_transaction_sheet_parse_or_score"] is True
    with pytest.raises(FileExistsError, match="receipt already exists"):
        fidelity_v6.run_entity_fidelity_v6_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=confirmation,
        )
    assert observed["loader_calls"] == 1


def test_receipt_readback_corruption_blocks_before_private_uci502_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    exploratory_path = tmp_path / "exploratory.json"
    attempt_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    final_path = tmp_path / "final.json"
    completion_path = tmp_path / "completion.json"
    exploratory_path.write_text("{}\n", encoding="utf-8")
    profile = _profile()
    monkeypatch.setitem(
        fidelity_v6.EXPECTED_PARTITION_HASHES,
        "calibration",
        profile.source_customer_ids_sha256,
    )
    protocol = {
        "protocol_sha256": "f" * 64,
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    for name, value in (
        ("DEFAULT_PROTOCOL", protocol_path),
        ("DEFAULT_EXPLORATORY", exploratory_path),
        ("DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path),
        ("DEFAULT_SEAL_RECEIPT", receipt_path),
        ("DEFAULT_OUTPUT", final_path),
        ("DEFAULT_COMPLETION_SEAL", completion_path),
    ):
        monkeypatch.setattr(fidelity_v6, name, value)
    monkeypatch.setattr(fidelity_v6, "load_frozen_v6_protocol", lambda _path: protocol)
    fidelity_v6._write_artifact(attempt_path, _attempt(protocol["protocol_sha256"]))
    exploratory_sha256 = hashlib.sha256(exploratory_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        fidelity_v6,
        "_recompute_and_match_exploratory",
        lambda *_args: (
            calibration_fixture(),
            {},
            {"gate_passed": True},
            exploratory_sha256,
        ),
    )
    monkeypatch.setattr(
        fidelity_v6, "fit_behavior_profile_v6", lambda *_args, **_kwargs: profile
    )
    raw_loader_calls = 0

    def forbidden_raw_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal raw_loader_calls
        raw_loader_calls += 1
        raise AssertionError("raw UCI502 loader crossed a corrupt receipt")

    monkeypatch.setattr(
        fidelity_v4, "load_retail_ii_readiness", forbidden_raw_loader
    )
    original_writer = fidelity_v6._write_artifact

    def corrupting_writer(path: Path, value: object) -> bytes:
        encoded = original_writer(path, value)
        if path == receipt_path:
            path.chmod(0o644)
            path.write_bytes(b"{}\n")
        return encoded

    monkeypatch.setattr(fidelity_v6, "_write_artifact", corrupting_writer)
    confirmation = (
        f"{fidelity_v6.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
    )
    with pytest.raises(ValueError, match="receipt bytes changed"):
        fidelity_v6.run_entity_fidelity_v6_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory_path,
            one_shot_final_confirmation=confirmation,
        )

    assert raw_loader_calls == 0
    assert receipt_path.exists() and not final_path.exists()
    assert not completion_path.exists()


def test_development_abort_writes_sanitized_immutable_stop_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "calibration.xlsx"
    protocol_path = tmp_path / "protocol.json"
    output = tmp_path / "exploratory.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(fidelity_v6, "DEFAULT_REFERENCE", reference)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_EXPLORATORY", output)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(
        fidelity_v6,
        "load_frozen_v6_protocol",
        lambda _path: {"protocol_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        fidelity_v6,
        "load_retail_partition",
        lambda *_args: (object(), object()),
    )
    monkeypatch.setattr(
        fidelity_v6,
        "_compute_calibration_exploratory",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("customer-secret")),
    )

    with pytest.raises(RuntimeError, match="customer-secret"):
        fidelity_v6.run_calibration_exploratory(
            reference, output, protocol_path=protocol_path
        )
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["status"] == "STOP_FAIL"
    assert failure["uci502_transactions_accessed"] is False
    assert failure["threshold_changed"] is False
    assert "customer-secret" not in json.dumps(failure)
    assert attempt.exists()
    assert oct(output.stat().st_mode & 0o777) == "0o444"


def test_final_recompute_failure_blocks_before_receipt_or_uci502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    exploratory = tmp_path / "exploratory.json"
    attempt = tmp_path / "attempt.json"
    receipt = tmp_path / "receipt.json"
    final = tmp_path / "final.json"
    completion = tmp_path / "completion.json"
    monkeypatch.setattr(fidelity_v6, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_EXPLORATORY", exploratory)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_SEAL_RECEIPT", receipt)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_OUTPUT", final)
    monkeypatch.setattr(fidelity_v6, "DEFAULT_COMPLETION_SEAL", completion)
    protocol = {"protocol_sha256": "b" * 64}
    monkeypatch.setattr(fidelity_v6, "load_frozen_v6_protocol", lambda _path: protocol)
    fidelity_v6._write_artifact(attempt, _attempt(protocol["protocol_sha256"]))
    monkeypatch.setattr(
        fidelity_v6,
        "_recompute_and_match_exploratory",
        lambda *_args: (_ for _ in ()).throw(ValueError("byte mismatch")),
    )
    calls = {"uci502": 0}
    monkeypatch.setattr(
        fidelity_v4,
        "load_retail_ii_readiness",
        lambda *_args, **_kwargs: calls.__setitem__("uci502", 1),
    )
    confirmation = f"{fidelity_v6.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
    with pytest.raises(ValueError, match="byte mismatch"):
        fidelity_v6.run_entity_fidelity_v6_final(
            protocol_path=protocol_path,
            exploratory_path=exploratory,
            one_shot_final_confirmation=confirmation,
        )
    assert calls["uci502"] == 0
    assert not receipt.exists() and not final.exists() and not completion.exists()


def test_completion_seal_controls_release_and_detects_exact_final_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    exploratory_path = tmp_path / "exploratory.json"
    attempt_path = tmp_path / "attempt.json"
    receipt_path = tmp_path / "receipt.json"
    final_path = tmp_path / "final.json"
    completion_path = tmp_path / "completion.json"
    final_reference = tmp_path / "uci502-never-opened.xlsx"
    for name, value in (
        ("DEFAULT_PROTOCOL", protocol_path),
        ("DEFAULT_EXPLORATORY", exploratory_path),
        ("DEFAULT_DEVELOPMENT_ATTEMPT", attempt_path),
        ("DEFAULT_SEAL_RECEIPT", receipt_path),
        ("DEFAULT_OUTPUT", final_path),
        ("DEFAULT_COMPLETION_SEAL", completion_path),
        ("DEFAULT_FINAL_REFERENCE", final_reference),
    ):
        monkeypatch.setattr(fidelity_v6, name, value)
    calibration = calibration_fixture()
    profile = _profile()
    protocol = {
        "protocol_sha256": "9" * 64,
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    monkeypatch.setattr(fidelity_v6, "load_frozen_v6_protocol", lambda _path: protocol)
    monkeypatch.setitem(
        fidelity_v6.EXPECTED_PARTITION_HASHES,
        "calibration",
        profile.source_customer_ids_sha256,
    )
    monkeypatch.setattr(
        fidelity_v6,
        "UCI352_CALIBRATION_CUSTOMERS",
        calibration["customer_id"].nunique(),
    )
    fidelity_v6._write_artifact(attempt_path, _attempt(protocol["protocol_sha256"]))
    exploratory = _exploratory_result(protocol["protocol_sha256"], passed=True)
    monkeypatch.setattr(fidelity_v6, "_exploratory_evidence_valid", lambda _value: True)
    exploratory_bytes = fidelity_v6._write_artifact(exploratory_path, exploratory)
    exploratory_sha256 = hashlib.sha256(exploratory_bytes).hexdigest()
    monkeypatch.setattr(
        fidelity_v6,
        "_recompute_and_match_exploratory",
        lambda *_args: (
            calibration,
            {},
            exploratory,
            exploratory_sha256,
        ),
    )
    monkeypatch.setattr(
        fidelity_v6, "fit_behavior_profile_v6", lambda *_args, **_kwargs: profile
    )
    minimum = fidelity_v6.UCI502_MIN_SPLIT_CUSTOMERS
    customer_ids = [f"historical-{index:04d}" for index in range(2 * minimum)]
    historical = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "ts": [
                pd.Timestamp("2009-12-01") + pd.Timedelta(minutes=index)
                for index in range(2 * minimum)
            ],
            "amount": [10.0 + index / 100 for index in range(2 * minimum)],
        }
    )
    transform_ids = set(customer_ids[:minimum])
    score_ids = set(customer_ids[minimum:])
    provenance = {
        "workbook_sha256": fidelity_v6.UCI502_WORKBOOK_SHA256,
        "manifest_sha256": fidelity_v6.UCI502_MANIFEST_SHA256,
        "sheet": fidelity_v6.UCI502_SHEET,
        "second_worksheet_rows_iterated": False,
        "nonselected_transactions_materialized": False,
    }
    monkeypatch.setattr(
        fidelity_v4,
        "load_retail_ii_readiness",
        lambda *_args, **_kwargs: (historical, provenance),
    )
    monkeypatch.setattr(
        fidelity_v6,
        "uci502_final_split",
        lambda _ids: (transform_ids, score_ids),
    )
    monkeypatch.setattr(
        fidelity_v6,
        "_final_seed_predictions",
        lambda *_args, **_kwargs: {
            "auc": 0.55,
            "d": 0.55,
            "real_customers": minimum,
            "synthetic_customers": minimum,
        },
    )
    monkeypatch.setattr(
        fidelity_v6,
        "_max_seed_bootstrap",
        lambda *_args, **_kwargs: ([0.55], 0.54, 0.56),
    )
    confirmation = (
        f"{fidelity_v6.ONE_SHOT_FINAL_CONFIRMATION}:{protocol['protocol_sha256']}"
    )
    result = fidelity_v6.run_entity_fidelity_v6_final(
        protocol_path=protocol_path,
        exploratory_path=exploratory_path,
        one_shot_final_confirmation=confirmation,
    )

    assert result["metrics"]["gate_passed"] is True
    assert oct(final_path.stat().st_mode & 0o777) == "0o444"
    assert oct(completion_path.stat().st_mode & 0o777) == "0o444"
    verification = fidelity_v6.verify_v6_artifacts(
        protocol_path=protocol_path,
        exploratory_path=exploratory_path,
        attempt_path=attempt_path,
        receipt_path=receipt_path,
        final_path=final_path,
        seal_path=completion_path,
    )
    assert verification["verified"] is True
    assert verification["checks"]["internal_completion_chain_valid"] is True
    assert verification["release_requires_external_signed_or_worm_custody"] is True

    poisoned_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in (
        "exploratory_artifact_sha256",
        "recomputed_exploratory_sha256",
        "behavior_profile_sha256",
    ):
        poisoned_receipt[key] = "customer-secret"
    poisoned_receipt["receipt_sha256"] = fidelity_v6._canonical_hash(
        {
            key: value
            for key, value in poisoned_receipt.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="values differ"):
        fidelity_v6._write_artifact(
            tmp_path / "poisoned-receipt.json", poisoned_receipt
        )

    final_scalar_leak_paths = (
        ("protocol",),
        ("bindings",),
        ("behavior_profile",),
        ("reference",),
        ("reference", "development_training"),
        ("reference", "one_shot_final"),
        ("seeds", str(fidelity_v6.EVALUATION_SEEDS[0])),
        ("metrics",),
    )
    for index, leak_path in enumerate(final_scalar_leak_paths):
        poisoned_final = json.loads(json.dumps(result))
        target = poisoned_final
        for key in leak_path:
            target = target[key]
        target["private_scalar"] = "customer-secret"
        with pytest.raises(ValueError, match="values differ"):
            fidelity_v6._write_artifact(
                tmp_path / f"poisoned-final-{index}.json", poisoned_final
            )
    assert not (tmp_path / "poisoned-receipt.json").exists()
    assert not any(tmp_path.glob("poisoned-final-*.json"))

    final_path.chmod(0o644)
    final_path.write_bytes(final_path.read_bytes() + b"\n")
    drifted = fidelity_v6.verify_v6_artifacts(
        protocol_path=protocol_path,
        exploratory_path=exploratory_path,
        attempt_path=attempt_path,
        receipt_path=receipt_path,
        final_path=final_path,
        seal_path=completion_path,
    )
    assert drifted["checks"]["final_valid"] is True
    assert drifted["checks"]["completion_seal_valid"] is False
    assert drifted["checks"]["internal_completion_chain_valid"] is False
    assert drifted["verified"] is False


@pytest.mark.parametrize(
    "path_name",
    ("protocol_path", "exploratory_path", "attempt_path", "receipt_path", "final_path", "seal_path"),
)
def test_verifier_rejects_every_noncanonical_resolved_path_before_read(
    path_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = {
        "protocol_path": tmp_path / "protocol.json",
        "exploratory_path": tmp_path / "exploratory.json",
        "attempt_path": tmp_path / "attempt.json",
        "receipt_path": tmp_path / "receipt.json",
        "final_path": tmp_path / "final.json",
        "seal_path": tmp_path / "completion.json",
    }
    for constant, argument in (
        ("DEFAULT_PROTOCOL", "protocol_path"),
        ("DEFAULT_EXPLORATORY", "exploratory_path"),
        ("DEFAULT_DEVELOPMENT_ATTEMPT", "attempt_path"),
        ("DEFAULT_SEAL_RECEIPT", "receipt_path"),
        ("DEFAULT_OUTPUT", "final_path"),
        ("DEFAULT_COMPLETION_SEAL", "seal_path"),
    ):
        monkeypatch.setattr(fidelity_v6, constant, canonical[argument])
    canonical[path_name] = tmp_path / "noncanonical.json"
    reads = 0

    def forbidden_protocol_read(_path: Path) -> object:
        nonlocal reads
        reads += 1
        raise AssertionError("protocol must not be read for noncanonical paths")

    monkeypatch.setattr(fidelity_v6, "load_frozen_v6_protocol", forbidden_protocol_read)
    with pytest.raises(ValueError, match="verification paths are canonical"):
        fidelity_v6.verify_v6_artifacts(**canonical)
    assert reads == 0


def test_verifier_rejects_symlink_alias_even_when_it_targets_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = {
        "protocol_path": tmp_path / "protocol.json",
        "exploratory_path": tmp_path / "exploratory.json",
        "attempt_path": tmp_path / "attempt.json",
        "receipt_path": tmp_path / "receipt.json",
        "final_path": tmp_path / "final.json",
        "seal_path": tmp_path / "completion.json",
    }
    for constant, argument in (
        ("DEFAULT_PROTOCOL", "protocol_path"),
        ("DEFAULT_EXPLORATORY", "exploratory_path"),
        ("DEFAULT_DEVELOPMENT_ATTEMPT", "attempt_path"),
        ("DEFAULT_SEAL_RECEIPT", "receipt_path"),
        ("DEFAULT_OUTPUT", "final_path"),
        ("DEFAULT_COMPLETION_SEAL", "seal_path"),
    ):
        monkeypatch.setattr(fidelity_v6, constant, canonical[argument])
    alias = tmp_path / "protocol-alias.json"
    alias.symlink_to(canonical["protocol_path"])
    canonical["protocol_path"] = alias
    with pytest.raises(ValueError, match="verification paths are canonical"):
        fidelity_v6.verify_v6_artifacts(**canonical)


def test_canonical_artifact_reader_rejects_symlinks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "regular.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "canonical-name.json"
    symlink.symlink_to(target)

    with pytest.raises(ValueError, match="regular non-symlink"):
        fidelity_v6._read_regular_file_bytes(symlink)
