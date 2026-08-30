from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import sim.population_v4 as population_v4

from sim.population_v4 import (
    QUANTILE_PROBABILITIES,
    STRATUM_CELL_COUNTS,
    STRATUM_ORDER,
    STRATUM_PRIMITIVES,
    BehaviorProfileV4,
    CheckerboardStratum,
    PopulationSimulatorV4,
    _largest_remainder_counts,
    checkerboard_marginals,
)


def _quantiles(name: str, primitive: str) -> tuple[float, ...]:
    if primitive == "invoice_count":
        value = 2.0 if name == "same_day_repeat" else 3.0
        return (value,) * 32
    if primitive == "active_start_fraction":
        return tuple(0.05 + 0.25 * index / 31 for index in range(32))
    if primitive == "active_duration_fraction":
        return tuple(0.15 + 0.25 * index / 31 for index in range(32))
    if primitive == "relative_log_amount_center":
        return tuple(-0.5 + index / 31 for index in range(32))
    if primitive == "eb_log_amount_scale":
        return tuple(0.2 + 0.4 * index / 31 for index in range(32))
    if primitive == "eb_short_session_probability":
        return tuple(0.1 + 0.5 * index / 31 for index in range(32))
    if primitive == "circadian_phase_latent":
        return tuple(2 * math.pi * index / 31 for index in range(32))
    if primitive == "eb_circadian_concentration":
        return tuple(0.5 + 2 * index / 31 for index in range(32))
    raise AssertionError(primitive)


def _profile() -> BehaviorProfileV4:
    strata = []
    for name in STRATUM_ORDER:
        dimensions = len(STRATUM_PRIMITIVES[name])
        masses = (1 / (3**dimensions),) * (3**dimensions)
        strata.append(
            CheckerboardStratum(
                name=name,
                customer_count=32,
                primitive_quantiles=tuple(
                    _quantiles(name, primitive)
                    for primitive in STRATUM_PRIMITIVES[name]
                ),
                cell_masses=masses,
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
        observation_days=365.0,
        session_gap_log_mean=math.log(3_600),
        session_gap_log_std=0.4,
        circadian_phase_cut=0.4,
        amount_variance_prior_alpha=1.2,
        amount_variance_prior_beta=0.25,
        short_session_prior_alpha=1.0,
        short_session_prior_beta=3.0,
    )


def test_checkerboard_shapes_marginals_and_order_independent_roundtrip() -> None:
    profile = _profile()
    payload = profile.as_dict()
    assert {
        name: len(payload["checkerboard"]["strata"][name]["cell_masses"])
        for name in STRATUM_ORDER
    } == STRATUM_CELL_COUNTS
    for stratum in profile.strata:
        assert math.isclose(sum(stratum.cell_masses), 1.0, abs_tol=1e-12)
        assert all(
            math.isclose(value, 1 / 3, abs_tol=1e-12)
            for marginal in checkerboard_marginals(
                stratum.cell_masses, len(stratum.primitives)
            )
            for value in marginal
        )
    reordered = json.loads(json.dumps(payload, sort_keys=True))
    restored = BehaviorProfileV4.from_dict(reordered)
    assert restored == profile
    assert restored.sha256 == profile.sha256


def test_largest_remainder_ties_use_declared_stratum_order() -> None:
    assert _largest_remainder_counts(2, (1, 1, 1)) == (1, 1, 0)
    assert _largest_remainder_counts(5, (1, 1, 1)) == (2, 2, 1)


def test_checkerboard_sampling_is_deterministic_and_exactly_supported() -> None:
    profile = _profile()
    first = profile.sample_primitives(99, seed=91)
    second = profile.sample_primitives(99, seed=91)
    assert first == second
    assert first != profile.sample_primitives(99, seed=92)
    singletons = [row for row in first if row["invoice_count"] == 1]
    same_day = [
        row
        for row in first
        if row["invoice_count"] >= 2 and row["active_duration_fraction"] == 0
    ]
    multiday = [
        row
        for row in first
        if row["invoice_count"] >= 2 and row["active_duration_fraction"] > 0
    ]
    assert tuple(map(len, (singletons, same_day, multiday))) == (33, 33, 33)
    assert all(row["fixed_hour"].is_integer() for row in singletons)
    assert all(row["eb_log_amount_scale"] > 0 for row in same_day + multiday)
    assert all(
        math.isclose(
            row["circadian_phase_sin"] ** 2 + row["circadian_phase_cos"] ** 2,
            1.0,
            abs_tol=1e-12,
        )
        for row in first
    )


def test_multiday_boundary_projects_start_before_positive_duration() -> None:
    profile = _profile()
    multiday = profile.strata[2]
    start_index = multiday.primitives.index("active_start_fraction")
    quantiles = list(multiday.primitive_quantiles)
    quantiles[start_index] = (1.0,) * 32
    boundary = replace(
        profile,
        strata=(
            profile.strata[0],
            profile.strata[1],
            replace(multiday, primitive_quantiles=tuple(quantiles)),
        ),
    )
    rows = [
        row
        for row in boundary.sample_primitives(96, seed=8)
        if row["invoice_count"] >= 2 and row["active_duration_fraction"] > 0
    ]
    assert rows
    assert all(row["active_start_fraction"] < 1 for row in rows)
    assert all(
        0 < row["active_duration_fraction"]
        and row["active_start_fraction"] + row["active_duration_fraction"] <= 1
        for row in rows
    )


def test_profile_is_fixed_size_and_rejects_hidden_raw_fields() -> None:
    payload = _profile().as_dict()
    serialized = json.dumps(payload, sort_keys=True)
    assert "customer_vectors" not in serialized
    assert "rank_matrix" not in serialized
    assert payload["privacy"]["contains_raw_rows"] is False
    assert payload["privacy"]["contains_evaluator_feature_rows"] is False
    payload["raw_rows"] = [{"customer_id": "leak"}]
    try:
        BehaviorProfileV4.from_dict(payload)
    except ValueError as exc:
        assert "fields differ" in str(exc)
    else:
        raise AssertionError("unexpected fields must fail closed")


def test_profile_rejects_nonfinite_renderer_and_prior_aggregates() -> None:
    profile = _profile()
    for field in (
        "observation_days",
        "session_gap_log_mean",
        "session_gap_log_std",
        "circadian_phase_cut",
        "amount_variance_prior_alpha",
        "amount_variance_prior_beta",
        "short_session_prior_alpha",
        "short_session_prior_beta",
    ):
        with pytest.raises(ValueError, match="renderer or empirical-Bayes"):
            replace(profile, **{field: float("nan")})


def test_v4_renderer_keeps_singletons_at_one_and_repeats_at_two_or_more() -> None:
    profile = _profile()
    simulator = PopulationSimulatorV4(
        seed=20260827,
        customers=96,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    events = list(simulator.iter_legitimate(192))
    counts: dict[str, int] = {}
    timestamps = []
    for event in events:
        customer_id = event["txn"]["customer_id"]
        counts[customer_id] = counts.get(customer_id, 0) + 1
        timestamps.append(event["ts"])
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "contracts/event.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for event in events:
        validator.validate(event)
    sampled = sorted(
        simulator.customers.values(), key=lambda customer: customer.customer_id
    )
    assert len(events) == 192
    assert timestamps == sorted(timestamps)
    assert all(
        counts[customer.customer_id] == 1
        for customer in sampled
        if round(customer.activity_weight) == 1
    )
    assert all(
        counts[customer.customer_id] >= 2
        for customer in sampled
        if round(customer.activity_weight) >= 2
    )


def test_v4_cli_loads_top_level_behavior_profile_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    profile_path = tmp_path / "final.json"
    profile_path.write_text(
        json.dumps({"behavior_profile": profile.as_dict()}, sort_keys=True),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate(output: Path, **kwargs: object) -> dict[str, object]:
        captured["output"] = output
        captured.update(kwargs)
        return {"events": kwargs["events"]}

    monkeypatch.setattr(population_v4, "generate_population_dataset_v4", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "population_v4",
            "--output",
            str(tmp_path / "events.jsonl"),
            "--events",
            "192",
            "--behavior-profile",
            str(profile_path),
        ],
    )
    population_v4.main()
    assert captured["events"] == 192
    assert isinstance(captured["behavior_profile"], BehaviorProfileV4)
    assert captured["behavior_profile"].sha256 == profile.sha256
