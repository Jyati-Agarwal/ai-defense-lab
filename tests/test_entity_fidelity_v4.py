from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sim.entity_fidelity_v4 as fidelity_v4
from sim.entity_fidelity_v4 import (
    NOVELTY_DRAWS,
    UCI502_HEADERS,
    UCI502_SHEET,
    _assert_aggregate_only,
    _crossfit_max_seed_bootstrap,
    _fit_behavior_profile_and_rows,
    _gate_metrics_valid,
    fractional_checkerboard_masses,
    fractional_tie_weights,
    load_retail_ii_readiness,
    novelty_check,
    generate_synthetic_transactions_v4,
    run_entity_fidelity_v4_final,
    run_temporal_readiness,
    uci502_readiness_split,
    verify_v4_artifacts,
)
from sim.population_v4 import (
    QUANTILE_PROBABILITIES,
    STRATUM_CELL_COUNTS,
    STRATUM_ORDER,
    STRATUM_PRIMITIVES,
    BehaviorProfileV4,
    CheckerboardStratum,
    PopulationSimulatorV4,
    checkerboard_marginals,
)


def _uniform_profile() -> BehaviorProfileV4:
    strata = []
    for name in STRATUM_ORDER:
        quantiles = []
        for primitive in STRATUM_PRIMITIVES[name]:
            if primitive == "invoice_count":
                value = 2.0 if name == "same_day_repeat" else 3.0
                values = (value,) * 32
            elif primitive == "active_start_fraction":
                values = tuple(0.05 + 0.2 * index / 31 for index in range(32))
            elif primitive == "active_duration_fraction":
                values = tuple(0.1 + 0.2 * index / 31 for index in range(32))
            elif primitive == "relative_log_amount_center":
                values = tuple(-0.3 + 0.6 * index / 31 for index in range(32))
            elif primitive == "eb_log_amount_scale":
                values = tuple(0.2 + 0.3 * index / 31 for index in range(32))
            elif primitive == "eb_short_session_probability":
                values = tuple(0.1 + 0.4 * index / 31 for index in range(32))
            elif primitive == "circadian_phase_latent":
                values = tuple(2 * math.pi * index / 31 for index in range(32))
            else:
                values = tuple(0.5 + index / 31 for index in range(32))
            quantiles.append(values)
        dimension = len(STRATUM_PRIMITIVES[name])
        strata.append(
            CheckerboardStratum(
                name=name,
                customer_count=32,
                primitive_quantiles=tuple(quantiles),
                cell_masses=(1 / 3**dimension,) * (3**dimension),
            )
        )
    return BehaviorProfileV4(
        source_dataset="fixture",
        source_partition="calibration",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=96,
        source_transaction_count=192,
        quantile_probabilities=QUANTILE_PROBABILITIES,
        strata=tuple(strata),
        hour_weights=(1 / 24,) * 24,
        calendar_weights=(1 / 12,) * 12,
        observation_days=365,
        session_gap_log_mean=math.log(3_600),
        session_gap_log_std=0.5,
        circadian_phase_cut=0.2,
        amount_variance_prior_alpha=1.2,
        amount_variance_prior_beta=0.25,
        short_session_prior_alpha=1,
        short_session_prior_beta=3,
    )


def _calibration_fixture() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2010-01-01")
    for index in range(32):
        day = base + pd.Timedelta(days=index)
        rows.append(
            {
                "customer_id": f"singleton-{index}",
                "ts": day + pd.Timedelta(hours=index % 24),
                "amount": 10.0 + index,
            }
        )
        rows.extend(
            [
                {
                    "customer_id": f"same-{index}",
                    "ts": day + pd.Timedelta(hours=(index + 3) % 20),
                    "amount": 20.0 + index,
                },
                {
                    "customer_id": f"same-{index}",
                    "ts": day + pd.Timedelta(hours=(index + 3) % 20 + 1),
                    "amount": 22.0 + index * 1.1,
                },
            ]
        )
        rows.extend(
            [
                {
                    "customer_id": f"multi-{index}",
                    "ts": day + pd.Timedelta(hours=index % 12),
                    "amount": 30.0 + index,
                },
                {
                    "customer_id": f"multi-{index}",
                    "ts": day + pd.Timedelta(hours=index % 12 + 1),
                    "amount": 34.0 + index * 1.2,
                },
                {
                    "customer_id": f"multi-{index}",
                    "ts": day + pd.Timedelta(days=2, hours=index % 12),
                    "amount": 39.0 + index * 1.4,
                },
            ]
        )
    return pd.DataFrame(rows).sort_values(["customer_id", "ts"], kind="stable")


def test_fractional_ties_and_checkerboard_have_exact_uniform_thirds() -> None:
    constant = fractional_tie_weights(np.ones(6))
    assert np.allclose(constant, np.full((6, 3), 1 / 3))
    tied = fractional_tie_weights(np.asarray([0, 0, 0, 0, 1, 1], dtype=float))
    assert np.allclose(tied.mean(axis=0), np.full(3, 1 / 3))
    rows = np.column_stack(
        (np.asarray([0, 0, 0, 0, 1, 1]), np.asarray([1, 1, 0, 0, 0, 0]))
    )
    masses = fractional_checkerboard_masses(rows)
    assert len(masses) == 9
    assert math.isclose(sum(masses), 1.0, abs_tol=1e-12)
    assert any(value == 0 for value in masses)
    assert all(
        math.isclose(value, 1 / 3, abs_tol=1e-12)
        for marginal in checkerboard_marginals(masses, 2)
        for value in marginal
    )


def test_fitted_v4_profile_is_deterministic_fixed_size_and_novel() -> None:
    calibration = _calibration_fixture()
    first, rows = _fit_behavior_profile_and_rows(
        calibration, customer_ids_sha256="c" * 64
    )
    second, _ = _fit_behavior_profile_and_rows(
        calibration, customer_ids_sha256="c" * 64
    )
    assert first == second
    assert first.sha256 == second.sha256
    assert {
        stratum.name: len(stratum.cell_masses) for stratum in first.strata
    } == STRATUM_CELL_COUNTS
    _assert_aggregate_only(first.as_dict())
    result = novelty_check(first, rows, draws=500, batch_size=100)
    assert result == {
        "draws": 500,
        "exact_full_continuous_primitive_tuple_matches": 0,
        "passed": True,
    }
    assert NOVELTY_DRAWS == 100_000


def test_fidelity_generation_preserves_every_sampled_invoice_count() -> None:
    profile = replace(_uniform_profile(), source_transaction_count=240)
    expected_simulator = PopulationSimulatorV4(
        seed=51,
        customers=96,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    expected = {
        customer.customer_id: round(customer.activity_weight)
        for customer in expected_simulator.customers.values()
    }
    generated = generate_synthetic_transactions_v4(
        profile=profile, customers=96, seed=51
    )
    actual = generated.groupby("customer_id").size().to_dict()
    assert actual == expected
    assert len(generated) == sum(expected.values())
    assert len(generated) != round(profile.activity_mean_transactions * 96)


def test_privacy_allowlist_rejects_rows_and_prediction_vectors() -> None:
    _assert_aggregate_only(_uniform_profile().as_dict())
    with pytest.raises(ValueError, match="not allowed"):
        _assert_aggregate_only({"feature_rows": [[1, 2, 3]]})
    with pytest.raises(ValueError, match="not allowed"):
        _assert_aggregate_only({"prediction_vectors": [0.1, 0.9]})
    with pytest.raises(ValueError, match="allowlisted aggregate container"):
        _assert_aggregate_only({"rows": [{"customer": "leak"}]})
    with pytest.raises(ValueError, match="allowlisted aggregate container"):
        _assert_aggregate_only({"scores": {"customer": 0.9}})


def test_gate_claim_cannot_override_failed_or_malformed_upper_bound() -> None:
    result = {
        "gate_passed": True,
        "metrics": {
            "bootstrap_max_seed_d_ci_95": [0.55, 0.64],
            "threshold_max_d_upper_95": 0.65,
        },
    }
    assert _gate_metrics_valid(result) is True
    result["metrics"]["bootstrap_max_seed_d_ci_95"] = [0.55, 0.66]
    assert _gate_metrics_valid(result) is False
    result["metrics"]["bootstrap_max_seed_d_ci_95"] = [0.55, float("nan")]
    assert _gate_metrics_valid(result) is False


def test_uci502_loader_accesses_named_first_sheet_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[str] = []
    rows = [
        UCI502_HEADERS,
        ("1", "x", "d", 1, pd.Timestamp("2010-01-02"), 4.0, 1.0, "UK"),
        ("2", "x", "d", 1, pd.Timestamp("2010-01-03"), 5.0, 2.0, "UK"),
        ("3", "x", "d", 1, pd.Timestamp("2010-12-02"), 6.0, 3.0, "UK"),
    ]

    class FakeSheet:
        def iter_rows(self, *, values_only: bool):
            assert values_only is True
            return iter(rows)

    class FakeWorkbook:
        sheetnames = [UCI502_SHEET, "Year 2010-2011"]

        def __getitem__(self, name: str):
            accesses.append(name)
            if name != UCI502_SHEET:
                raise AssertionError("second sheet must never be opened")
            return FakeSheet()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        fidelity_v4,
        "verify_retail_ii_provenance",
        lambda _path: {
            "workbook_sha256": "w",
            "manifest_sha256": "m",
        },
    )
    monkeypatch.setattr(fidelity_v4, "load_workbook", lambda *_args, **_kwargs: FakeWorkbook())
    monkeypatch.setattr(fidelity_v4, "UCI502_EXPECTED_ROWS", 3)
    frame, reference = load_retail_ii_readiness(
        Path("unused.xlsx"), excluded_customer_ids={"2"}
    )
    assert accesses == [UCI502_SHEET]
    assert frame[["customer_id", "amount"]].to_dict("records") == [
        {"customer_id": "1", "amount": 4.0}
    ]
    assert reference["training_customer_line_items_excluded"] == 1
    assert reference["outside_fixed_window_line_items"] == 1
    assert reference["second_worksheet_rows_iterated"] is False


def test_uci502_provenance_failure_precedes_workbook_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fidelity_v4,
        "verify_retail_ii_provenance",
        lambda _path: (_ for _ in ()).throw(ValueError("bad pinned bytes")),
    )
    monkeypatch.setattr(
        fidelity_v4,
        "load_workbook",
        lambda *_args, **_kwargs: pytest.fail("unverified workbook must not open"),
    )
    with pytest.raises(ValueError, match="bad pinned bytes"):
        load_retail_ii_readiness(Path("tampered.xlsx"), excluded_customer_ids=set())


def test_uci502_hash_split_is_deterministic_disjoint_and_exhaustive() -> None:
    customer_ids = {f"customer-{index}" for index in range(100)}
    first = uci502_readiness_split(customer_ids, minimum_customers=1)
    second = uci502_readiness_split(reversed(sorted(customer_ids)), minimum_customers=1)
    assert first == second
    assert first[0].isdisjoint(first[1])
    assert first[0] | first[1] == customer_ids


def test_crossfit_bootstrap_is_deterministic_and_shares_real_population() -> None:
    seed_folds = {}
    for seed in (1, 2):
        folds = []
        for fold in range(5):
            folds.append(
                {
                    "fold": fold,
                    "_real_customer_ids": (
                        f"customer-{fold}-0",
                        f"customer-{fold}-1",
                    ),
                    "_real_predictions": np.asarray([0.1, 0.2]),
                    "_synthetic_predictions": np.asarray([0.8, 0.9]),
                }
            )
        seed_folds[seed] = folds
    first = _crossfit_max_seed_bootstrap(seed_folds, repeats=100, seed=7)
    second = _crossfit_max_seed_bootstrap(seed_folds, repeats=100, seed=7)
    assert first == second
    assert first[2] >= first[1] >= 0.99


def test_failed_exploratory_blocks_all_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"protocol_sha256": "frozen"}
    exploratory = tmp_path / "exploratory.json"
    exploratory.write_text(
        json.dumps(
            {
                "protocol_sha256": "frozen",
                "gate_passed": False,
                "calibration_only": True,
                "diagnostic_transactions_accessed": False,
                "uci502_transactions_accessed": False,
                "sealed_final_transactions_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fidelity_v4, "load_frozen_v4_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        fidelity_v4,
        "load_retail_partition",
        lambda *_args, **_kwargs: pytest.fail("no UCI352 partition may be opened"),
    )
    monkeypatch.setattr(
        fidelity_v4,
        "load_retail_ii_readiness",
        lambda *_args, **_kwargs: pytest.fail("UCI502 must remain unopened"),
    )
    with pytest.raises(ValueError, match="exploratory gate"):
        run_temporal_readiness(
            tmp_path / "uci352.xlsx",
            tmp_path / "uci502.xlsx",
            tmp_path / "readiness.json",
            protocol_path=tmp_path / "protocol.json",
            exploratory_path=exploratory,
        )


def test_wrong_sealed_confirmation_precedes_any_partition_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fidelity_v4,
        "load_frozen_v4_protocol",
        lambda _path: {"protocol_sha256": "frozen"},
    )
    monkeypatch.setattr(
        fidelity_v4,
        "load_retail_partition",
        lambda *_args, **_kwargs: pytest.fail("no partition may be opened"),
    )
    with pytest.raises(ValueError, match="protocol-hash-bound"):
        run_entity_fidelity_v4_final(
            tmp_path / "reference.xlsx",
            tmp_path / "final.json",
            protocol_path=tmp_path / "protocol.json",
            readiness_path=tmp_path / "readiness.json",
            receipt_path=tmp_path / "receipt.json",
            sealed_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:wrong",
        )


def test_receipt_is_exclusive_and_written_before_sealed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _uniform_profile()
    protocol = {
        "protocol_sha256": "frozen",
        "generator": {"behavior_profile_sha256": profile.sha256},
    }
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    calls: list[str] = []
    calibration = _calibration_fixture()

    def fake_partition(_path: Path, partition: str):
        calls.append(partition)
        if partition == "calibration":
            return calibration, {}
        assert partition == "sealed_final"
        assert receipt_path.exists()
        raise RuntimeError("fake sealed boundary reached")

    monkeypatch.setattr(fidelity_v4, "load_frozen_v4_protocol", lambda _path: protocol)
    monkeypatch.setattr(fidelity_v4, "_load_passed_readiness", lambda *_args: {})
    monkeypatch.setattr(fidelity_v4, "fit_behavior_profile_v4", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(fidelity_v4, "load_retail_partition", fake_partition)
    with pytest.raises(RuntimeError, match="fake sealed boundary"):
        run_entity_fidelity_v4_final(
            tmp_path / "reference.xlsx",
            tmp_path / "final.json",
            protocol_path=tmp_path / "protocol.json",
            readiness_path=readiness_path,
            receipt_path=receipt_path,
            sealed_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:frozen",
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert calls == ["calibration", "sealed_final"]
    assert "diagnostic" not in calls
    assert receipt["status"] == "consumed"
    assert receipt["protocol_sha256"] == "frozen"
    with pytest.raises(FileExistsError, match="receipt already exists"):
        run_entity_fidelity_v4_final(
            tmp_path / "reference.xlsx",
            tmp_path / "final.json",
            protocol_path=tmp_path / "protocol.json",
            readiness_path=readiness_path,
            receipt_path=receipt_path,
            sealed_final_confirmation="APPROVED_SINGLE_FINAL_SCORE:frozen",
        )
    assert calls == ["calibration", "sealed_final"]


def test_verify_fails_closed_without_sealed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fidelity_v4,
        "load_frozen_v4_protocol",
        lambda _path: {"protocol_sha256": "frozen"},
    )
    result = verify_v4_artifacts(
        protocol_path=tmp_path / "protocol.json",
        exploratory_path=tmp_path / "exploratory.json",
        readiness_path=tmp_path / "readiness.json",
        receipt_path=tmp_path / "receipt.json",
        final_path=tmp_path / "final.json",
    )
    assert result["verified"] is False
    assert result["checks"] == {
        "exploratory_exists": False,
        "readiness_exists": False,
        "sealed_artifacts_complete": False,
    }
