from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

import sim.population_v6 as population_v6
from sim.population_v4 import _largest_remainder_counts
from sim.population_v6 import (
    CHECKERBOARD_STRATA,
    QUANTILE_PROBABILITIES,
    REPEAT_PRIMITIVES,
    SINGLETON_CONTINUOUS_PRIMITIVES,
    STRATUM_CELL_COUNTS,
    SAMPLED_PRIMITIVE_FIELDS,
    BehaviorProfileV6,
    CheckerboardStratumV6,
    PopulationSimulatorV6,
    generate_population_dataset_v6,
    moment_matched_hour_counts,
)


def _quantiles(name: str) -> tuple[float, ...]:
    if name == "invoice_count":
        return (3.0,) * 32
    if name in {
        "active_day_position",
        "active_end_fraction",
        "span_within_end_fraction",
    }:
        return tuple(0.05 + 0.90 * index / 31 for index in range(32))
    if name == "relative_log_amount_center":
        return tuple(-0.2 + 0.4 * index / 31 for index in range(32))
    if name == "eb_log_amount_scale":
        return tuple(0.2 + 0.4 * index / 31 for index in range(32))
    if name == "eb_short_session_probability":
        return tuple(0.1 + 0.4 * index / 31 for index in range(32))
    if name == "circadian_phase_latent":
        return tuple(2 * math.pi * index / 32 for index in range(32))
    if name == "circadian_resultant":
        return tuple(0.1 + 0.8 * index / 31 for index in range(32))
    raise AssertionError(name)


def profile_fixture() -> BehaviorProfileV6:
    singleton = CheckerboardStratumV6(
        name="singleton",
        customer_count=32,
        primitive_quantiles=tuple(
            _quantiles(name) for name in SINGLETON_CONTINUOUS_PRIMITIVES
        ),
        cell_masses=(1 / STRATUM_CELL_COUNTS["singleton"],)
        * STRATUM_CELL_COUNTS["singleton"],
    )
    repeat = CheckerboardStratumV6(
        name="multiday_repeat",
        customer_count=32,
        primitive_quantiles=tuple(_quantiles(name) for name in REPEAT_PRIMITIVES),
        cell_masses=(1 / STRATUM_CELL_COUNTS["multiday_repeat"],)
        * STRATUM_CELL_COUNTS["multiday_repeat"],
    )
    return BehaviorProfileV6(
        source_dataset="synthetic fixture",
        source_partition="calibration",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=96,
        source_transaction_count=224,
        repeat_customer_count=64,
        same_day_repeat_probability=0.5,
        quantile_probabilities=QUANTILE_PROBABILITIES,
        strata=(singleton, repeat),
        hour_weights=(1 / 24,) * 24,
        calendar_weights=(1 / 12,) * 12,
        observation_days=30,
        session_gap_log_mean=math.log(60),
        session_gap_log_std=0.4,
        circadian_phase_cut=0.4,
        global_log_amount_center=math.log(250),
        localization_factor=1.0,
        amount_variance_prior_alpha=1.2,
        amount_variance_prior_beta=0.25,
        short_session_prior_alpha=1.0,
        short_session_prior_beta=3.0,
    )


def test_v6_schema_roundtrip_shapes_privacy_and_amount_calibration() -> None:
    profile = profile_fixture()
    payload = profile.as_dict()

    assert payload["schema_version"] == 6
    assert {
        name: len(payload["checkerboard"]["strata"][name]["cell_masses"])
        for name in CHECKERBOARD_STRATA
    } == STRATUM_CELL_COUNTS
    assert payload["checkerboard"]["strata"]["singleton"]["cell_shape"] == [3, 3, 24]
    assert payload["checkerboard"]["strata"]["multiday_repeat"]["cell_shape"] == [3] * 8
    assert payload["renderer"]["global_log_amount_center"] == math.log(250)
    assert payload["renderer"]["localization_factor"] == 1.0
    assert payload["privacy"]["contains_prediction_vectors"] is False
    assert BehaviorProfileV6.from_dict(json.loads(json.dumps(payload))) == profile

    hostile = json.loads(json.dumps(payload))
    hostile["raw_rows"] = [{"customer_id": "forbidden"}]
    with pytest.raises(ValueError, match="fields differ"):
        BehaviorProfileV6.from_dict(hostile)

    noninteger_count = json.loads(json.dumps(payload))
    noninteger_count["checkerboard"]["strata"]["multiday_repeat"][
        "primitive_quantiles"
    ]["invoice_count"][0] = 2.5
    with pytest.raises(ValueError, match="repeat quantiles"):
        BehaviorProfileV6.from_dict(noninteger_count)

    subminor_center = json.loads(json.dumps(payload))
    subminor_center["renderer"]["global_log_amount_center"] = -100
    with pytest.raises(ValueError, match="below one minor unit"):
        BehaviorProfileV6.from_dict(subminor_center)


def test_v6_sampling_decodes_integer_support_without_clipping() -> None:
    profile = profile_fixture()
    rows = profile.sample_primitives(96, seed=91)
    singletons = [row for row in rows if row["invoice_count"] == 1]
    repeats = [row for row in rows if row["invoice_count"] >= 2]
    atoms = [row for row in repeats if row["same_day_atom"] == 1]
    multiday = [row for row in repeats if row["same_day_atom"] == 0]

    assert rows == profile.sample_primitives(96, seed=91)
    assert rows != profile.sample_primitives(96, seed=92)
    assert tuple(map(len, (singletons, atoms, multiday))) == (32, 32, 32)
    assert all(row["first_day"] == row["last_day"] for row in singletons + atoms)
    assert all(
        0 <= row["first_day"] < row["last_day"] < profile.observation_days
        for row in multiday
    )
    assert all(row["invoice_count"] == 3 for row in repeats)
    assert all(set(row) == SAMPLED_PRIMITIVE_FIELDS for row in rows)
    assert all(row["fixed_hour"].is_integer() and 0 <= row["fixed_hour"] < 24 for row in singletons)
    assert _largest_remainder_counts(96, (32, 64)) == (32, 64)


def test_repeat_hour_tilt_is_exact_integer_and_empirical_base_bounded() -> None:
    counts, moment = moment_matched_hour_counts(
        (1 / 24,) * 24, phase=1.17, resultant=0.72, count=37
    )
    points = [
        (math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24))
        for hour in range(24)
    ]

    assert len(counts) == 24
    assert sum(counts) == 37
    assert all(count >= 0 for count in counts)
    assert math.isclose(moment[0], sum(counts[h] * points[h][0] for h in range(24)) / 37, abs_tol=1e-12)
    assert math.isclose(moment[1], sum(counts[h] * points[h][1] for h in range(24)) / 37, abs_tol=1e-12)

    one_hour, one_moment = moment_matched_hour_counts(
        (0.0,) * 7 + (1.0,) + (0.0,) * 16,
        phase=0.0,
        resultant=0.0,
        count=4,
    )
    assert one_hour[7] == 4
    assert math.isclose(one_moment[0], points[7][0], abs_tol=1e-12)


def test_two_event_repair_escapes_one_event_local_minimum() -> None:
    phase = 0.178149
    resultant = 0.120957
    counts, moment = moment_matched_hour_counts(
        (1 / 24,) * 24, phase=phase, resultant=resultant, count=2
    )
    target = (
        resultant * math.sin(phase),
        resultant * math.cos(phase),
    )
    selected = [hour for hour, value in enumerate(counts) for _ in range(value)]
    squared_error = sum((actual - expected) ** 2 for actual, expected in zip(moment, target))

    assert selected == [6, 19]
    assert squared_error < 0.001


def test_sparse_boundary_hour_repair_terminates_deterministically() -> None:
    weights = (
        0.0, 0.0360910478926936, 0.012247279355618961, 0.0,
        0.046020174964454146, 0.06830640528488689, 0.05368180223097496, 0.0,
        0.0, 0.08611559294175887, 0.0848141268737954, 0.05330493983295846,
        0.0, 0.059014286956243875, 0.007230297283579116, 0.04952090376005941,
        0.01669504327905661, 0.08795964510272694, 0.04232911734777521,
        0.07123205030049021, 0.08523443075928012, 0.0,
        0.0938079643192697, 0.04639489151437765,
    )
    first = moment_matched_hour_counts(
        weights, phase=0.18418394490058815, resultant=1.0, count=37
    )
    second = moment_matched_hour_counts(
        weights, phase=0.18418394490058815, resultant=1.0, count=37
    )

    assert first == second
    assert sum(first[0]) == 37
    assert all(value >= 0 for value in first[0])


def test_projected_boundary_uses_exact_limiting_face_distribution() -> None:
    weights_list = [0.0] * 24
    for hour, value in {
        0: 0.2536767870316542,
        2: 0.1645945315118449,
        18: 0.3289131389901339,
        19: 0.1958673207575056,
        23: 0.056948221708861366,
    }.items():
        weights_list[hour] = value
    counts, moment = moment_matched_hour_counts(
        tuple(weights_list),
        phase=0.5426361321872425,
        resultant=1e-12,
        count=5,
    )

    assert sum(counts) == 5
    assert all(value >= 0 for value in counts)
    assert all(math.isfinite(value) for value in moment)


def test_randomized_sparse_boundary_hour_repair_has_no_abort() -> None:
    rng = random.Random(20260827)
    for _ in range(24):
        support = sorted(rng.sample(range(24), rng.choice((2, 3, 5, 7, 12, 20))))
        raw = [rng.random() + 0.01 for _ in support]
        total = sum(raw)
        weights = [0.0] * 24
        for hour, value in zip(support, raw, strict=True):
            weights[hour] = value / total
        count = rng.choice((2, 3, 5, 7, 17, 37))
        counts, moment = moment_matched_hour_counts(
            weights,
            phase=2 * math.pi * rng.random(),
            resultant=rng.choice((0.0, 1e-12, 0.25, 1.0)),
            count=count,
        )

        assert sum(counts) == count
        assert all(value >= 0 for value in counts)
        assert all(math.isfinite(value) for value in moment)


def test_v6_renderer_preserves_exact_counts_days_hours_and_source_amount_scale() -> None:
    profile = profile_fixture()
    simulator = PopulationSimulatorV6(
        seed=23,
        customers=12,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    target = round(profile.activity_mean_transactions * 12)
    expected = {
        customer.customer_id: round(customer.activity_weight)
        for customer in simulator.customers.values()
    }
    events = list(simulator.iter_legitimate(target))
    observed = Counter(event["txn"]["customer_id"] for event in events)
    timestamps: dict[str, list[datetime]] = defaultdict(list)
    for event in events:
        timestamps[event["txn"]["customer_id"]].append(
            datetime.fromisoformat(event["ts"].replace("Z", "+00:00"))
        )

    assert len(events) == target == 28
    assert observed == expected
    assert max(customer.avg_ticket_minor for customer in simulator.customers.values()) < 40_000
    for customer in simulator.customers.values():
        values = timestamps[customer.customer_id]
        assert len(values) == expected[customer.customer_id]
        assert len(values) == len(set(values))
        assert values == sorted(values)
        assert (values[0] - simulator._profile_start).days == customer._v6_first_day
        assert (values[-1] - simulator._profile_start).days == customer._v6_last_day
        assert (len({value.date() for value in values}) == 1) is (
            customer._v6_first_day == customer._v6_last_day
        )
        assert all(
            (right - left).total_seconds() > 86_400
            for left, right in zip(values, values[1:])
            if left.date() != right.date()
        )


def test_positive_multiday_support_spans_dates_at_fractional_edge() -> None:
    profile = profile_fixture()
    simulator = PopulationSimulatorV6(
        seed=29,
        customers=3,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    customer = next(iter(simulator.customers.values()))
    customer.activity_weight = 2
    customer.fixed_hour = None
    customer._v6_first_day = 0
    customer._v6_last_day = 1
    customer._v6_phase = 0.0
    customer._v6_resultant = math.nextafter(0.5, math.inf)
    schedule = simulator._customer_schedule(customer, 2)

    assert len(schedule) == 2
    assert len({timestamp.date() for timestamp, _ in schedule}) == 2


def test_adjacent_day_same_hour_sessions_are_strictly_more_than_one_day_apart() -> None:
    profile = replace(
        profile_fixture(),
        hour_weights=(1.0,) + (0.0,) * 23,
        session_gap_log_mean=math.log(86_400),
        session_gap_log_std=0.0,
    )
    simulator = PopulationSimulatorV6(
        seed=291,
        customers=3,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    customer = next(iter(simulator.customers.values()))
    customer.activity_weight = 10
    customer.fixed_hour = None
    customer.short_session_probability = 1.0
    customer._v6_first_day = 0
    customer._v6_last_day = 1
    customer._v6_phase = 0.0
    customer._v6_resultant = 1.0
    schedule = simulator._customer_schedule(customer, 10)
    timestamps = [timestamp for timestamp, _ in schedule]

    assert len(timestamps) == 10
    assert {timestamp.hour for timestamp in timestamps} == {0}
    assert len({timestamp.date() for timestamp in timestamps}) == 2
    boundary = next(
        index
        for index, (left, right) in enumerate(zip(timestamps, timestamps[1:]))
        if left.date() != right.date()
    )
    assert (timestamps[boundary + 1] - timestamps[boundary]).total_seconds() > 86_400


def test_v4_minimum_preserving_allocator_keeps_v6_strata_for_residual_target() -> None:
    simulator = PopulationSimulatorV6(
        seed=30,
        customers=12,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile_fixture(),
    )
    events = list(simulator.iter_legitimate(29))
    counts = Counter(event["txn"]["customer_id"] for event in events)

    assert len(events) == 29
    assert all(
        counts[customer.customer_id] == 1
        for customer in simulator.customers.values()
        if round(customer.activity_weight) == 1
    )
    assert all(
        counts[customer.customer_id] >= 2
        for customer in simulator.customers.values()
        if round(customer.activity_weight) >= 2
    )


def test_same_day_capacity_is_fail_closed() -> None:
    simulator = PopulationSimulatorV6(
        seed=31,
        customers=3,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile_fixture(),
    )
    customer = next(iter(simulator.customers.values()))
    customer._v6_first_day = customer._v6_last_day = 2
    with pytest.raises(ValueError, match="one-second capacity"):
        simulator._customer_schedule(customer, 86_401)


def test_canonical_delayed_attack_is_sorted_without_clock_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SpySimulator(PopulationSimulatorV6):
        last: "SpySimulator | None" = None

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.clock_history = [self.clock]
            type(self).last = self

        def generate_legitimate(self, **kwargs: object) -> dict[str, object]:
            event = super().generate_legitimate(**kwargs)
            self.clock_history.append(self.clock)
            return event

    monkeypatch.setattr(population_v6, "PopulationSimulatorV6", SpySimulator)
    output = tmp_path / "events.jsonl"
    manifest = generate_population_dataset_v6(
        output,
        events=24,
        seed=20260827,
        fraud_rate=0.5,
        benign_adversarial_rate=0,
        behavior_profile=profile_fixture(),
    )
    events = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    timestamps = [
        datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00"))
        for event in events
    ]
    simulator = SpySimulator.last

    assert simulator is not None
    assert manifest["events"] == len(events) == 24
    assert any(
        event["label"]["attack_family"] == "ttl_dormancy" for event in events
    )
    assert timestamps == sorted(timestamps)
    assert simulator.clock_history == sorted(simulator.clock_history)
