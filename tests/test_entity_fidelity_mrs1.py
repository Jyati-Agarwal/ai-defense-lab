from __future__ import annotations

import inspect
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sim.entity_fidelity_mrs1 as mrs1
from sim.entity_fidelity import _ids_hash, _transactions_hash
from sim.population_mrs1 import CalendarHazardClock, MRS1Profile, generate_synthetic_transactions_mrs1
from tests.test_population_mrs1 import profile_fixture


def _reference(partition: str, households: int, baskets: int) -> dict[str, object]:
    return {
        "dataset": "completejourney",
        "version": "1.1.1",
        "partition": partition,
        "manifest_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
        "household_ids_sha256": "e" * 64,
        "households": households,
        "baskets": baskets,
        "source_rds_sha256": "f" * 64,
        "mrs_preregistration_sha256": "b" * 64,
        "timestamp_semantics": mrs1.COMPLETEJOURNEY_TIMESTAMP_SEMANTICS,
        "observation_start": mrs1.COMPLETEJOURNEY_OBSERVATION_START,
        "observation_end_exclusive": mrs1.COMPLETEJOURNEY_OBSERVATION_END,
        "amount_unit": "USD cents",
        "observation_end": mrs1.COMPLETEJOURNEY_OBSERVATION_END,
    }


def _isolate_lifecycle_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filenames = {
        "DEFAULT_PREREGISTRATION": "preregistration.json",
        "DEFAULT_PROTOCOL": "protocol.json",
        "DEFAULT_QUALIFICATION_RECEIPT": "qualification-receipt.json",
        "DEFAULT_QUALIFICATION": "qualification.json",
        "DEFAULT_UCI352_TRANSFER": "transfer.json",
        "DEFAULT_FINAL_RECEIPT": "final-receipt.json",
        "DEFAULT_FINAL": "final.json",
        "DEFAULT_COMPLETION_SEAL": "completion-seal.json",
    }
    for name, filename in filenames.items():
        monkeypatch.setattr(mrs1, name, tmp_path / filename)
    monkeypatch.setattr(
        mrs1,
        "DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR",
        tmp_path / "completejourney",
    )


def test_calendar_fit_uses_unique_epochs_jeffreys_and_mean_one() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "a", "a", "b"],
            "ts": pd.to_datetime(
                [
                    "2017-01-02 03:00",
                    "2017-01-02 03:00",
                    "2017-01-02 04:00",
                    "2017-01-03 03:00",
                ]
            ),
            "amount": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fitted = mrs1.fit_calendar_hazard(
        frame,
        observation_start="2017-01-02T00:00:00",
        observation_end="2017-01-04T00:00:00",
    )

    assert fitted["hour_counts"][3] == 2
    assert fitted["hour_counts"][4] == 1
    assert sum(fitted["hour_counts"]) == 3
    assert fitted["hour_weights"][3] / fitted["hour_weights"][4] == pytest.approx(
        (2 + 0.5) / (1 + 0.5)
    )
    assert np.mean(fitted["hour_weights"]) == pytest.approx(1)
    assert np.mean(fitted["weekday_weights"]) == pytest.approx(1)
    assert np.mean(fitted["week_weights"]) == pytest.approx(1)
    assert all(value > 0 for value in fitted["week_weights"])


def test_2017_unobserved_week53_has_neutral_transfer_weight() -> None:
    start = pd.Timestamp("2017-01-01T00:00:00")
    end = pd.Timestamp("2018-01-01T00:00:00")
    timestamps = pd.date_range(start, end, freq="h", inclusive="left")
    intensity = 7
    frame = pd.DataFrame(
        {
            "customer_id": [
                f"exposure-probe-{customer}"
                for _timestamp in timestamps
                for customer in range(intensity)
            ],
            "ts": np.repeat(timestamps.to_numpy(), intensity),
        }
    )

    fitted = mrs1.fit_calendar_hazard(
        frame,
        observation_start=start,
        observation_end=end,
    )

    assert fitted["week_counts"][52] == 0
    assert fitted["week_exposure_hours"][52] == 0
    assert fitted["week_weights"][52] == pytest.approx(1.0)
    assert all(math.isfinite(value) and value > 0 for value in fitted["week_weights"])


def test_zero_truncated_censored_timing_likelihood_and_quadrature_are_finite() -> None:
    profile = profile_fixture()
    clock = CalendarHazardClock(
        profile,
        datetime.fromisoformat(profile.observation_start),
        datetime.fromisoformat(profile.observation_end),
    )
    sequences = [
        ("one", np.asarray([0.2, 1.1, 4.0])),
        ("two", np.asarray([8.0])),
    ]
    value = mrs1.timing_negative_log_likelihood(
        np.log([1.5, 2.0, 2.5]), sequences, horizon=clock.total
    )
    frailties, log_weights = mrs1._quadrature_log_weights(2.5)

    assert math.isfinite(value)
    assert value != mrs1.timing_negative_log_likelihood(
        np.log([1.5, 2.0, 2.5]), sequences, horizon=clock.total + 2
    )
    assert np.exp(log_weights).sum() == pytest.approx(1)
    assert np.sum(np.exp(log_weights) * frailties) == pytest.approx(1, abs=0.02)


def test_end_to_end_fit_keeps_posteriors_ephemeral_and_is_deterministic() -> None:
    source_profile = profile_fixture()
    frame = generate_synthetic_transactions_mrs1(
        source_profile, customers=60, seed=20260827
    )
    arguments = {
        "source_dataset": "synthetic-fit-check",
        "source_partition": "development",
        "customer_ids_sha256": _ids_hash(frame["customer_id"].unique()),
        "transactions_sha256": _transactions_hash(frame),
        "observation_start": source_profile.observation_start,
        "observation_end": source_profile.observation_end,
    }
    first = mrs1.fit_mrs1_profile(frame, **arguments)
    second = mrs1.fit_mrs1_profile(frame, **arguments)

    assert first == second
    assert first.sha256 == second.sha256
    assert 0.25 < first.renewal_shape < 10
    assert 0.05 < first.mean_operational_gap_days < 365
    assert 0.25 < first.frailty_shape < 20
    assert -0.95 < first.amount_ar1 < 0.95
    assert first.amount_innovation_scale > 0
    encoded = json.dumps(first.as_dict())
    assert '"posterior_weights":' not in encoded
    assert '"customer_id":' not in encoded


def test_strict_optimizer_stops_on_nonconvergence_and_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False, fun=1.0, x=np.asarray([0.5])
        ),
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        mrs1._strict_minimize(lambda _value: 1.0, ((0.5,),), ((0.0, 1.0),), label="test")

    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True, fun=1.0, x=np.asarray([0.0])
        ),
    )
    with pytest.raises(RuntimeError, match="boundary"):
        mrs1._strict_minimize(lambda _value: 1.0, ((0.5,),), ((0.0, 1.0),), label="test")


def test_gate_evidence_rejects_coercible_types() -> None:
    folds = [
        {
            "fold": fold,
            "training_real_customers": 40,
            "evaluation_real_customers": 10,
            "evaluation_synthetic_customers": 10,
            "training_customer_ids_sha256": "a" * 64,
            "evaluation_customer_ids_sha256": "b" * 64,
            "profile_sha256": "c" * 64,
            "auc": 0.6,
            "d": 0.6,
        }
        for fold in range(5)
    ]
    evidence = {
        "seeds": {
            str(seed): {"fold_weighted_d": 0.6, "folds": folds}
            for seed in mrs1.EVALUATION_SEEDS
        },
        "metrics": {
            "orientation_free_max_seed_d": 0.6,
            "bootstrap_max_seed_d_mean": 0.6,
            "bootstrap_max_seed_d_ci_95": [0.55, 0.64],
            "threshold_max_d_upper_95": 0.65,
        },
        "gate_passed": True,
    }
    assert mrs1._gate_evidence_valid(evidence)
    evidence["seeds"][str(mrs1.EVALUATION_SEEDS[0])]["folds"][0][
        "training_real_customers"
    ] = "40"
    assert not mrs1._gate_evidence_valid(evidence)


def test_completejourney_adapter_matches_exact_loader_api_and_fixed_window() -> None:
    households = [f"h-{index}" for index in range(mrs1.DEVELOPMENT_HOUSEHOLDS)]
    raw = pd.DataFrame(
        {
            "household_id": households,
            "basket_id": [f"b-{index}" for index in range(len(households))],
            "transaction_timestamp": pd.to_datetime(
                ["2017-06-01 12:00"] * len(households)
            ),
            "sales_value_cents": np.arange(1, len(households) + 1, dtype="int64"),
        }
    )
    source = _reference("development", len(households), len(raw))

    def loader(
        output_dir: Path,
        partition: str,
        *,
        expected_manifest_sha256: str,
        expected_mrs_preregistration_sha256: str,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        assert output_dir == Path("reference")
        assert partition == "development"
        assert expected_manifest_sha256 == "c" * 64
        assert expected_mrs_preregistration_sha256 == "b" * 64
        return raw, source

    frame, provenance = mrs1.load_completejourney_for_mrs1(
        Path("reference"),
        "development",
        expected_manifest_sha256="c" * 64,
        expected_mrs_preregistration_sha256="b" * 64,
        loader=loader,
    )

    assert tuple(frame.columns) == ("customer_id", "ts", "amount")
    assert frame["amount"].iloc[0] == 0.01
    assert provenance["observation_start"] == "2017-01-01T00:00:00"
    assert provenance["observation_end"] == "2018-01-01T00:00:00"
    assert provenance["profile_customer_ids_sha256"] == _ids_hash(households)


def test_r2_preregistration_binds_stopped_attempts_and_freezes_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle_paths(tmp_path, monkeypatch)
    path = mrs1.DEFAULT_PREREGISTRATION
    r0_sha256 = mrs1._file_sha256(mrs1.SUPERSEDED_PREREGISTRATION)
    r1_sha256 = mrs1._file_sha256(mrs1.R1_PREREGISTRATION)
    value = mrs1.write_mrs1_preregistration()

    assert value["artifact"] == "entity_fidelity_mrs1_r2_preregistration"
    assert value["study_revision"] == "mrs1.2"
    assert value["lineage"]["r0_to_r1"]["preflight_failure_sha256"] == (
        "10f4cba07c6d37f6012c04a81d0468409a38dc26334774bdfb34cb8223b2aca9"
    )
    assert value["lineage"]["r0_to_r1"]["superseded_preregistration_sha256"] == (
        "648c0d1d8020c06c6d184cf451c9caff90acd401e6db3b62393b91439ce3cb7a"
    )
    assert value["lineage"]["r1_preregistration_sha256"] == (
        "24c080955926b7c4ea42ef8d590f3b4e8250bc755ddbb25b6a90385a95357b2e"
    )
    assert value["lineage"]["r1_preprocess_failure_sha256"] == (
        "9aa113dbaea1b72f8b1cd0cae8a6a167290fabcc1e008f4f2b6289e01c941099"
    )
    assert mrs1._file_sha256(mrs1.SUPERSEDED_PREREGISTRATION) == r0_sha256
    assert mrs1._file_sha256(mrs1.R1_PREREGISTRATION) == r1_sha256
    assert value["source_bytes_downloaded"] is True
    assert value["source_rds_bytes_read_for_sha256_verification"] is True
    assert value["source_transaction_rows_transiently_parsed"] is True
    assert value["source_transaction_rows_persisted_or_retained"] is False
    assert value["basket_aggregation_completed_in_memory"] is True
    assert value["basket_aggregation_persisted_or_retained"] is False
    assert value["completejourney_partitions_materialized_or_loaded"] is False
    assert value["access_scope"] == {
        "canonical_mrs_lifecycle_is_single_use": True,
        "arbitrary_same_user_python_reads_cryptographically_prevented": False,
        "admin_readable_raw_partitions_require_external_worm_custody": True,
    }
    assert value["new_model_family"] is True
    assert value["not_v10_tuning"] is True
    assert value["model"]["calendar_hazard_shape"] == [24, 7, 53]
    assert value["optimizer"]["gauss_laguerre_nodes"] == 48
    assert value["evaluator"]["seeds"] == [20260827, 20260828, 20260829]
    assert value["evaluator"]["folds"] == 5
    assert value["evaluator"]["bootstrap_repeats"] == 10_000
    assert value["evaluator"]["orientation_free_threshold"] == 0.65
    assert value["evaluator"]["development_fit_diagnostic"]["optimistic"] is True
    assert value["evaluator"]["development_fit_diagnostic"]["out_of_sample"] is False
    assert value["evaluator"]["qualification_is_first_independent_release_gate"] is True
    assert value["completejourney"]["source_chain"] == mrs1._reference_source_chain()
    assert value["completejourney"]["partition_manifest_state"] == (
        "not_created_after_stopped_r1_preprocessing"
    )
    assert value["completejourney"]["partition_manifest_sha256"] is None
    assert value["completejourney"]["source_timestamp_timezone"] == (
        "America/New_York"
    )
    assert value["completejourney"]["stored_timestamp_semantics"] == (
        mrs1.COMPLETEJOURNEY_TIMESTAMP_SEMANTICS
    )
    assert value["completejourney"]["source_chain"]["rds_sha256"] == (
        "1fa0700033f1e5d9bb6b09e2be063d8d68474d346e95c50f2833e09d083e0007"
    )
    assert oct(path.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(FileExistsError):
        mrs1.write_mrs1_preregistration()


def test_r2_preregistration_rejects_changed_r0_preflight_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle_paths(tmp_path, monkeypatch)
    changed_path = tmp_path / "changed-preflight-failure.json"
    changed = json.loads(mrs1.DEFAULT_PREFLIGHT_FAILURE.read_bytes())
    changed["uv_reported_version"] = "uv 0.11.9 (x86_64-unknown-linux-gnu)"
    changed_path.write_bytes(mrs1._canonical_bytes(changed))
    changed_path.chmod(0o444)
    monkeypatch.setattr(mrs1, "DEFAULT_PREFLIGHT_FAILURE", changed_path)

    with pytest.raises(ValueError, match="preflight-failure evidence changed"):
        mrs1.write_mrs1_preregistration()
    assert not mrs1.DEFAULT_PREREGISTRATION.exists()


def test_r2_preregistration_rejects_changed_r1_preprocess_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle_paths(tmp_path, monkeypatch)
    changed_path = tmp_path / "changed-r1-preprocess-failure.json"
    changed = json.loads(mrs1.R1_PREPROCESS_FAILURE.read_bytes())
    changed["deterministic_household_split_computed"] = True
    changed_path.write_bytes(mrs1._canonical_bytes(changed))
    changed_path.chmod(0o444)
    monkeypatch.setattr(mrs1, "R1_PREPROCESS_FAILURE", changed_path)

    with pytest.raises(ValueError, match="r1 preprocessing-failure evidence changed"):
        mrs1.write_mrs1_preregistration()
    assert not mrs1.DEFAULT_PREREGISTRATION.exists()


def test_freeze_checks_manifest_back_binding_before_development_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle_paths(tmp_path, monkeypatch)
    preregistration_path = mrs1.DEFAULT_PREREGISTRATION
    mrs1.write_mrs1_preregistration()
    reference_dir = mrs1.DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
    reference_dir.mkdir(parents=True)
    manifest_path = reference_dir / "completejourney_reference.manifest.json"
    manifest_path.write_bytes(
        mrs1._canonical_bytes({"mrs_preregistration_sha256": "0" * 64})
    )
    manifest_path.chmod(0o444)
    import sim.completejourney_reference as reference

    monkeypatch.setattr(reference, "_validate_manifest", lambda value: value)
    called = False

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("development partition loader must not run")

    monkeypatch.setattr(mrs1, "load_completejourney_for_mrs1", forbidden_loader)
    with pytest.raises(ValueError, match="not bound"):
        mrs1.freeze_mrs1_protocol()
    assert called is False


def test_preregistration_refuses_backdating_after_partition_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle_paths(tmp_path, monkeypatch)
    reference_dir = mrs1.DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
    reference_dir.mkdir(parents=True)
    (reference_dir / "completejourney_development.parquet").write_bytes(b"marker")

    with pytest.raises(FileExistsError, match="downstream materialization"):
        mrs1.write_mrs1_preregistration()
    assert not mrs1.DEFAULT_PREREGISTRATION.exists()


def test_freeze_checks_full_preregistered_source_chain_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_lifecycle_paths(tmp_path, monkeypatch)
    preregistration = mrs1.write_mrs1_preregistration()
    preregistration_sha256 = mrs1.hashlib.sha256(
        mrs1._canonical_bytes(preregistration)
    ).hexdigest()
    source = dict(preregistration["completejourney"]["source_chain"])
    source["rds_sha256"] = "0" * 64
    manifest = {
        "mrs_preregistration_sha256": preregistration_sha256,
        "source": {
            "source_preregistration_sha256": source[
                "source_preregistration_sha256"
            ],
            "quarantine_receipt_sha256": source["quarantine_receipt_sha256"],
            "manifest_sha256": source["rds_manifest_sha256"],
            "rds_sha256": source["rds_sha256"],
        },
    }
    reference_dir = mrs1.DEFAULT_COMPLETEJOURNEY_REFERENCE_DIR
    reference_dir.mkdir(parents=True)
    manifest_path = reference_dir / "completejourney_reference.manifest.json"
    manifest_path.write_bytes(mrs1._canonical_bytes(manifest))
    manifest_path.chmod(0o444)
    import sim.completejourney_reference as reference

    monkeypatch.setattr(reference, "_validate_manifest", lambda value: value)
    called = False

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("development partition loader must not run")

    monkeypatch.setattr(mrs1, "load_completejourney_for_mrs1", forbidden_loader)
    with pytest.raises(ValueError, match="source chain"):
        mrs1.freeze_mrs1_protocol()
    assert called is False


def test_failed_development_diagnostic_stops_before_qualification_receipt_or_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "qualification-receipt.json"
    output_path = tmp_path / "qualification.json"
    monkeypatch.setattr(mrs1, "DEFAULT_PROTOCOL", tmp_path / "protocol.json")
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION_RECEIPT", receipt_path)
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION", output_path)
    monkeypatch.setattr(
        mrs1,
        "load_frozen_mrs1_protocol",
        lambda _path: {"development_fit_diagnostic": {"gate_passed": False}},
    )
    called = False

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("qualification loader must not run")

    monkeypatch.setattr(mrs1, "load_completejourney_for_mrs1", forbidden_loader)
    with pytest.raises(RuntimeError, match="development-fit diagnostic"):
        mrs1.run_mrs1_qualification()
    assert called is False
    assert not receipt_path.exists() and not output_path.exists()


def test_production_one_shot_entrypoints_do_not_accept_artifact_paths() -> None:
    for function in (
        mrs1.write_mrs1_preregistration,
        mrs1.freeze_mrs1_protocol,
        mrs1.run_mrs1_qualification,
        mrs1.run_mrs1_frozen_transfer,
        mrs1.run_mrs1_one_shot_final,
    ):
        parameters = inspect.signature(function).parameters
        assert "output_path" not in parameters
        assert "receipt_path" not in parameters
        assert "loader" not in parameters
        assert "reference_dir" not in parameters


def test_qualification_receipt_exists_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    receipt_path = tmp_path / "qualification-receipt.json"
    output_path = tmp_path / "qualification.json"
    protocol = {
        "protocol_sha256": "a" * 64,
        "expected_completejourney_manifest_sha256": "c" * 64,
        "mrs_preregistration_sha256": "b" * 64,
        "development_profile": profile_fixture().as_dict(),
        "development_fit_diagnostic": {"gate_passed": True},
    }
    reference = _reference("qualification", mrs1.QUALIFICATION_HOUSEHOLDS, 1)
    frame = pd.DataFrame(
        {
            "customer_id": ["one"],
            "ts": [pd.Timestamp("2017-06-01")],
            "amount": [1.0],
        }
    )
    monkeypatch.setattr(mrs1, "load_frozen_mrs1_protocol", lambda _path: protocol)
    monkeypatch.setattr(mrs1, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION_RECEIPT", receipt_path)
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION", output_path)

    def load(*_args: object, **_kwargs: object) -> tuple[pd.DataFrame, dict[str, object]]:
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["receipt_before_qualification_partition_load_or_use"] is True
        return frame, reference

    monkeypatch.setattr(mrs1, "load_completejourney_for_mrs1", load)
    monkeypatch.setattr(
        mrs1,
        "evaluate_mrs1_gate",
        lambda *_args, **_kwargs: {
            "seeds": {},
            "metrics": {"bootstrap_max_seed_d_ci_95": [0.7, 0.8]},
            "gate_passed": False,
        },
    )
    result = mrs1.run_mrs1_qualification()

    assert result["status"] == "STOP_FAIL"
    assert receipt_path.exists() and output_path.exists()
    with pytest.raises(FileExistsError, match="consumed"):
        mrs1.run_mrs1_qualification()


def test_crossfit_renders_frozen_profile_in_target_reference_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = profile_fixture()
    rows = []
    for index in range(20):
        rows.extend(
            (
                {
                    "customer_id": f"c-{index}",
                    "ts": pd.Timestamp("2010-01-02") + pd.Timedelta(days=offset),
                    "amount": 1.0 + offset,
                }
                for offset in (0, 1)
            )
        )
    frame = pd.DataFrame(rows)
    observed: dict[str, object] = {}

    def generate(*_args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        raise RuntimeError("generation reached")

    monkeypatch.setattr(mrs1, "generate_synthetic_transactions_mrs1", generate)
    with pytest.raises(RuntimeError, match="generation reached"):
        mrs1._crossfit_seed(
            frame,
            {
                "partition": "transfer",
                "observation_start": "2010-01-01T00:00:00",
                "observation_end": "2010-02-01T00:00:00",
            },
            seed=20260827,
            frozen_profile=profile,
        )

    assert observed["start"] == datetime(2010, 1, 1)
    assert observed["end"] == datetime(2010, 2, 1)


def test_uci352_transfer_blocks_unpinned_source_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_text("{}")
    profile = profile_fixture()
    monkeypatch.setattr(
        mrs1,
        "load_passed_qualification",
        lambda _path: {"qualified_profile": profile.as_dict()},
    )
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION", qualification_path)
    monkeypatch.setattr(
        mrs1, "DEFAULT_UCI352_TRANSFER", tmp_path / "transfer.json"
    )
    monkeypatch.setattr(
        mrs1,
        "evaluate_mrs1_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("gate must not run")
        ),
    )

    monkeypatch.setattr(
        mrs1,
        "load_uci352_for_mrs1",
        lambda _path: (
            pd.DataFrame(
                {
                    "customer_id": ["x"],
                    "ts": [pd.Timestamp("2011-01-01")],
                    "amount": [1.0],
                }
            ),
            {**mrs1._uci352_transfer_source(), "workbook_sha256": "0" * 64},
        ),
    )
    with pytest.raises(ValueError, match="provenance"):
        mrs1.run_mrs1_frozen_transfer()


def test_self_authored_qualification_cannot_unlock_uci352(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_bytes(
        mrs1._canonical_bytes(
            {
                "artifact": "entity_fidelity_mrs1_qualification",
                "gate_passed": True,
                "status": "PASS",
            }
        )
    )
    qualification_path.chmod(0o444)
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION", qualification_path)
    monkeypatch.setattr(mrs1, "DEFAULT_PROTOCOL", tmp_path / "missing-protocol.json")
    monkeypatch.setattr(mrs1, "DEFAULT_UCI352_TRANSFER", tmp_path / "transfer.json")
    opened = False

    def forbidden_loader(_path: Path) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("UCI352 must remain unopened")

    monkeypatch.setattr(mrs1, "load_uci352_for_mrs1", forbidden_loader)
    with pytest.raises(FileNotFoundError):
        mrs1.run_mrs1_frozen_transfer()
    assert opened is False


def test_private_uci502_loader_rejects_bad_prerequisite_before_workbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "final-receipt.json"
    transfer_path = tmp_path / "transfer.json"
    transfer = {"artifact": "synthetic-transfer-snapshot"}
    transfer_path.write_bytes(mrs1._canonical_bytes(transfer))
    transfer_path.chmod(0o444)
    qualification = {"protocol_sha256": "a" * 64}
    confirmation = f"{mrs1.FINAL_CONFIRMATION}:{'a' * 64}"
    receipt = {
        "schema_version": mrs1.SCHEMA_VERSION,
        "artifact": "entity_fidelity_mrs1_one_shot_final_receipt",
        "status": "consumed",
        "protocol_sha256": "a" * 64,
        "prerequisite_artifact_sha256": "0" * 64,
        "confirmation_sha256": mrs1.hashlib.sha256(confirmation.encode()).hexdigest(),
        "final_source": mrs1._uci502_final_source(),
        "receipt_before_uci502_workbook_load_or_use": True,
    }
    receipt["receipt_sha256"] = mrs1._canonical_hash(receipt)
    receipt_path.write_bytes(mrs1._canonical_bytes(receipt))
    receipt_path.chmod(0o444)
    monkeypatch.setattr(mrs1, "DEFAULT_FINAL_RECEIPT", receipt_path)
    monkeypatch.setattr(mrs1, "DEFAULT_UCI352_TRANSFER", transfer_path)
    monkeypatch.setattr(
        mrs1, "load_passed_qualification", lambda _path: qualification
    )
    monkeypatch.setattr(
        mrs1,
        "_validate_passed_transfer_value",
        lambda value, _qualification: value,
    )
    opened = False

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("UCI502 workbook must remain unopened")

    import sim.entity_fidelity_v4 as fidelity_v4

    monkeypatch.setattr(fidelity_v4, "load_retail_ii_readiness", forbidden_loader)
    with pytest.raises(ValueError, match="receipt"):
        mrs1._load_uci502_final_for_mrs1()
    assert opened is False


def test_uci502_loader_runs_only_after_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_path = tmp_path / "qualification.json"
    transfer_path = tmp_path / "transfer.json"
    receipt_path = tmp_path / "final-receipt.json"
    output_path = tmp_path / "final.json"
    seal_path = tmp_path / "seal.json"
    qualification_path.write_text("{}")
    transfer_path.write_text(
        json.dumps({"gate_passed": True, "status": "PASS"}, indent=2, sort_keys=True)
        + "\n"
    )
    transfer_path.chmod(0o444)
    profile = profile_fixture()
    qualification = {
        "protocol_sha256": "a" * 64,
        "qualified_profile": profile.as_dict(),
        "qualified_profile_sha256": profile.sha256,
        "gate_passed": True,
        "status": "PASS",
    }
    monkeypatch.setattr(mrs1, "load_passed_qualification", lambda _path: qualification)
    monkeypatch.setattr(
        mrs1,
        "_validate_passed_transfer_value",
        lambda _value, _qualification: {
            "gate_passed": True,
            "status": "PASS",
            "qualification_sha256": mrs1.hashlib.sha256(
                mrs1._canonical_bytes(qualification)
            ).hexdigest(),
            "qualified_profile_sha256": profile.sha256,
        },
    )
    monkeypatch.setattr(mrs1, "DEFAULT_QUALIFICATION", qualification_path)
    monkeypatch.setattr(mrs1, "DEFAULT_UCI352_TRANSFER", transfer_path)
    monkeypatch.setattr(mrs1, "DEFAULT_FINAL_RECEIPT", receipt_path)
    monkeypatch.setattr(mrs1, "DEFAULT_FINAL", output_path)
    monkeypatch.setattr(mrs1, "DEFAULT_COMPLETION_SEAL", seal_path)
    monkeypatch.setattr(
        mrs1,
        "evaluate_mrs1_gate",
        lambda *_args, **_kwargs: {
            "seeds": {},
            "metrics": {"bootstrap_max_seed_d_ci_95": [0.7, 0.8]},
            "gate_passed": False,
        },
    )

    def loader() -> tuple[pd.DataFrame, dict[str, object]]:
        receipt_path = mrs1.DEFAULT_FINAL_RECEIPT
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["artifact"] == "entity_fidelity_mrs1_one_shot_final_receipt"
        frame = pd.DataFrame(
            {
                "customer_id": ["one"],
                "ts": [pd.Timestamp("2010-01-02")],
                "amount": [1.0],
            }
        )
        return frame, {
            "dataset": "uci502",
            **mrs1._uci502_final_source(),
            "partition": "one-shot-final",
            "observation_start": mrs1.UCI502_WINDOW_START,
            "observation_end": mrs1.UCI502_WINDOW_END,
        }

    monkeypatch.setattr(mrs1, "_load_uci502_final_for_mrs1", loader)
    result = mrs1.run_mrs1_one_shot_final(
        confirmation=f"{mrs1.FINAL_CONFIRMATION}:{'a' * 64}",
    )

    assert result["status"] == "STOP_FAIL"
    assert receipt_path.exists() and output_path.exists()
    assert not seal_path.exists()
