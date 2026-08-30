from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

import sim.population_v5 as population_v5
from sim.population_v4 import checkerboard_marginals
from sim.population_v5 import (
    ATOM_SELECTION_NAMESPACE,
    ATOM_SELECTION_RULE,
    CHECKERBOARD_STRATA,
    QUANTILE_PROBABILITIES,
    STRATUM_CELL_COUNTS,
    STRATUM_PRIMITIVES,
    BehaviorProfileV5,
    CheckerboardStratum,
    PopulationSimulatorV5,
    _largest_remainder_counts,
    generate_population_dataset_v5,
)


def _quantiles(primitive: str) -> tuple[float, ...]:
    if primitive == "invoice_count":
        return (3.0,) * 32
    if primitive == "active_start_fraction":
        return tuple(0.05 + 0.25 * index / 31 for index in range(32))
    if primitive == "active_duration_fraction":
        return tuple(0.10 + 0.25 * index / 31 for index in range(32))
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


def _profile() -> BehaviorProfileV5:
    strata = tuple(
        CheckerboardStratum(
            name=name,
            customer_count=32,
            primitive_quantiles=tuple(
                _quantiles(primitive) for primitive in STRATUM_PRIMITIVES[name]
            ),
            cell_masses=(1 / STRATUM_CELL_COUNTS[name],)
            * STRATUM_CELL_COUNTS[name],
        )
        for name in CHECKERBOARD_STRATA
    )
    return BehaviorProfileV5(
        source_dataset="synthetic fixture",
        source_partition="calibration",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=96,
        source_transaction_count=224,
        repeat_customer_count=64,
        same_day_repeat_probability=0.5,
        quantile_probabilities=QUANTILE_PROBABILITIES,
        strata=strata,
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


def test_v5_schema_roundtrip_privacy_and_no_dp_claim() -> None:
    profile = _profile()
    payload = profile.as_dict()

    assert payload["schema_version"] == 5
    assert payload["checkerboard"]["smoothing"] is False
    assert payload["checkerboard"]["pseudocounts"] is False
    assert payload["checkerboard"]["stored_rank_rows"] is False
    assert payload["same_day_repeat_atom"]["selection"] == ATOM_SELECTION_RULE
    assert payload["privacy"] == {
        "fixed_size_aggregate": True,
        "contains_customer_ids": False,
        "contains_raw_rows": False,
        "contains_rank_rows": False,
        "contains_evaluator_feature_rows": False,
        "differential_privacy_claimed": False,
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert not any(
        token in serialized
        for token in ("customer_vectors", "rank_matrix", '"raw_rows"', '"epsilon"')
    )

    restored = BehaviorProfileV5.from_dict(json.loads(serialized))
    assert restored == profile
    assert restored.sha256 == profile.sha256

    leaked = json.loads(serialized)
    leaked["raw_rows"] = [{"customer_id": "must-not-be-stored"}]
    with pytest.raises(ValueError, match="fields differ"):
        BehaviorProfileV5.from_dict(leaked)

    false_dp_claim = json.loads(serialized)
    false_dp_claim["privacy"]["differential_privacy_claimed"] = True
    with pytest.raises(ValueError, match="privacy declaration"):
        BehaviorProfileV5.from_dict(false_dp_claim)


def test_v5_checkerboard_masses_have_uniform_thirds_and_stable_ties() -> None:
    profile = _profile()

    assert {
        stratum.name: len(stratum.cell_masses) for stratum in profile.strata
    } == STRATUM_CELL_COUNTS
    for stratum in profile.strata:
        assert math.isclose(sum(stratum.cell_masses), 1.0, abs_tol=1e-12)
        assert all(
            math.isclose(mass, 1 / 3, abs_tol=1e-12)
            for marginal in checkerboard_marginals(
                stratum.cell_masses, len(stratum.primitives)
            )
            for mass in marginal
        )

    assert _largest_remainder_counts(1, (1, 1)) == (1, 0)
    assert _largest_remainder_counts(3, (1, 1)) == (2, 1)


def test_v5_sampling_is_deterministic_and_respects_support() -> None:
    profile = _profile()
    first = profile.sample_primitives(96, seed=91)

    assert first == profile.sample_primitives(96, seed=91)
    assert first != profile.sample_primitives(96, seed=92)
    singletons = [row for row in first if row["invoice_count"] == 1]
    repeats = [row for row in first if row["invoice_count"] >= 2]
    same_day = [row for row in repeats if row["active_duration_fraction"] == 0]
    multiday = [row for row in repeats if row["active_duration_fraction"] > 0]

    assert tuple(map(len, (singletons, same_day, multiday))) == (32, 32, 32)
    assert all(
        row["fixed_hour"].is_integer()
        and 0 <= row["fixed_hour"] < 24
        and row["eb_log_amount_scale"] == 0
        and row["eb_short_session_probability"] == 0
        and row["eb_circadian_concentration"] == 0
        for row in singletons
    )
    assert all(
        row["invoice_count"] == 3
        and row["eb_log_amount_scale"] > 0
        and 0 <= row["eb_short_session_probability"] <= 1
        and row["eb_circadian_concentration"] >= 0
        for row in repeats
    )
    assert all(
        0 <= row["active_start_fraction"] <= 1
        and 0 < row["active_duration_fraction"]
        and row["active_start_fraction"] + row["active_duration_fraction"] <= 1
        for row in multiday
    )
    assert all(
        math.isclose(
            row["circadian_phase_sin"] ** 2 + row["circadian_phase_cos"] ** 2,
            1.0,
            abs_tol=1e-12,
        )
        for row in first
    )


def test_v5_same_day_atom_uses_exact_declared_hash_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()

    def positioned_rows(
        _self: BehaviorProfileV5,
        stratum: CheckerboardStratum,
        count: int,
        _rng: object,
    ) -> list[dict[str, float]]:
        if stratum.name == "singleton":
            return [
                {"invoice_count": 1.0, "active_duration_fraction": 0.0}
                for _ in range(count)
            ]
        return [
            {
                "invoice_count": 3.0,
                "active_duration_fraction": 0.25,
                "position": float(index),
            }
            for index in range(count)
        ]

    monkeypatch.setattr(BehaviorProfileV5, "_sample_checkerboard", positioned_rows)
    seed = 47
    rows = profile.sample_primitives(10, seed=seed)
    repeats = [row for row in rows if row["invoice_count"] >= 2]
    expected_count = round(profile.same_day_repeat_probability * len(repeats))
    expected_positions = set(
        sorted(
            range(len(repeats)),
            key=lambda index: (
                hashlib.sha256(
                    f"{seed}:{ATOM_SELECTION_NAMESPACE}:{index}".encode()
                ).digest(),
                index,
            ),
        )[:expected_count]
    )
    selected_positions = {
        int(row["position"])
        for row in repeats
        if row["active_duration_fraction"] == 0
    }

    assert len(repeats) == 7
    assert selected_positions == expected_positions


def test_population_simulator_v5_preserves_sampled_invoice_counts() -> None:
    simulator = PopulationSimulatorV5(
        seed=19,
        customers=12,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=_profile(),
    )
    expected = {
        customer.customer_id: round(customer.activity_weight)
        for customer in simulator.customers.values()
    }
    events = list(simulator.iter_legitimate(sum(expected.values())))
    observed = Counter(event["txn"]["customer_id"] for event in events)

    assert set(expected.values()) == {1, 3}
    assert dict(observed) == expected
    assert [event["ts"] for event in events] == sorted(
        event["ts"] for event in events
    )


def test_rendered_same_day_atom_is_exact_and_strictly_chronological() -> None:
    profile = _profile()
    simulator = PopulationSimulatorV5(
        seed=23,
        customers=96,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    expected_counts = {
        customer.customer_id: round(customer.activity_weight)
        for customer in simulator.customers.values()
    }
    repeat_ids = {
        customer.customer_id
        for customer in simulator.customers.values()
        if round(customer.activity_weight) >= 2
    }
    atom_ids = {
        customer.customer_id
        for customer in simulator.customers.values()
        if round(customer.activity_weight) >= 2
        and customer.active_duration_fraction == 0
    }
    events = list(simulator.iter_legitimate(sum(expected_counts.values())))
    timestamps: dict[str, list[datetime]] = defaultdict(list)
    for event in events:
        timestamps[event["txn"]["customer_id"]].append(
            datetime.fromisoformat(event["ts"].replace("Z", "+00:00"))
        )
    observed_same_day = {
        customer_id
        for customer_id in repeat_ids
        if len({timestamp.date() for timestamp in timestamps[customer_id]}) == 1
    }

    assert Counter(event["txn"]["customer_id"] for event in events) == expected_counts
    assert len(repeat_ids) == 64
    assert len(atom_ids) == round(profile.same_day_repeat_probability * len(repeat_ids))
    assert observed_same_day == atom_ids
    assert all(
        all(right > left for left, right in zip(values, values[1:]))
        for values in timestamps.values()
    )


def test_canonical_v5_generator_uses_v5_scheduler_and_renders_exact_atom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SpySimulator(PopulationSimulatorV5):
        last: "SpySimulator | None" = None

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            type(self).last = self

    monkeypatch.setattr(population_v5, "PopulationSimulatorV5", SpySimulator)
    output = tmp_path / "events.jsonl"
    manifest = generate_population_dataset_v5(
        output,
        events=96,
        seed=29,
        fraud_rate=0,
        benign_adversarial_rate=0,
        behavior_profile=_profile(),
    )
    simulator = SpySimulator.last
    assert simulator is not None
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    timestamps: dict[str, list[datetime]] = defaultdict(list)
    for event in events:
        timestamps[event["txn"]["customer_id"]].append(
            datetime.fromisoformat(event["ts"].replace("Z", "+00:00"))
        )
    repeat_ids = {
        customer.customer_id
        for customer in simulator.customers.values()
        if round(customer.activity_weight) >= 2
    }
    atom_ids = {
        customer.customer_id
        for customer in simulator.customers.values()
        if round(customer.activity_weight) >= 2
        and customer.active_duration_fraction == 0
    }
    observed_same_day = {
        customer_id
        for customer_id in repeat_ids
        if len({timestamp.date() for timestamp in timestamps[customer_id]}) == 1
    }

    assert manifest["events"] == len(events) == 96
    assert manifest["behavior_profile_schema_version"] == 5
    assert observed_same_day == atom_ids
