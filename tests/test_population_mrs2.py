from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta

import pandas as pd
import pytest

from sim.entity_fidelity_mrs1 import fit_calendar_hazard
from sim.population_mrs2 import (
    MRS1_STOP_PROTOCOL_SHA256,
    CalendarHazardClock,
    MRS2Profile,
    MarkedRenewalSimulator,
    generate_synthetic_transactions_mrs2,
    pooled_hour_probabilities,
    positive_half_even_cents,
    preferred_hour_multiplier,
    sequence_sha256,
)


def profile_fixture_mrs2(
    *, multiplicity_values: tuple[int, ...] = (1,), preference_strength: float = 0.7
) -> MRS2Profile:
    start = datetime(2025, 1, 1, 0, 17)
    end = datetime(2025, 1, 4, 5, 11)
    epochs = pd.DataFrame(
        {
            "customer_id": [f"source-{index % 24}" for index in range(72)],
            "ts": [start + timedelta(hours=index, minutes=3) for index in range(72)],
        }
    )
    calendar = fit_calendar_hazard(
        epochs, observation_start=start, observation_end=end
    )
    counts = tuple(72 if index == 0 else 0 for index in range(len(multiplicity_values)))
    probabilities = tuple(count / 72 for count in counts)
    transactions = sum(
        value * count
        for value, count in zip(multiplicity_values, counts, strict=True)
    )
    g = pooled_hour_probabilities(
        start=start,
        end=end,
        hour_weights=calendar["hour_weights"],
        weekday_weights=calendar["weekday_weights"],
        week_weights=calendar["week_weights"],
    )
    return MRS2Profile(
        source_dataset="synthetic-unit-fixture",
        source_partition="structural",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=24,
        source_transaction_count=transactions,
        observation_start=start.isoformat(),
        observation_end=end.isoformat(),
        hour_counts=calendar["hour_counts"],
        weekday_counts=calendar["weekday_counts"],
        week_counts=calendar["week_counts"],
        hour_exposure_hours=calendar["hour_exposure_hours"],
        weekday_exposure_hours=calendar["weekday_exposure_hours"],
        week_exposure_hours=calendar["week_exposure_hours"],
        hour_weights=calendar["hour_weights"],
        weekday_weights=calendar["weekday_weights"],
        week_weights=calendar["week_weights"],
        preferred_hour_probabilities=g,
        preferred_hour_strength=preference_strength,
        renewal_shape=1.8,
        mean_operational_gap_days=1.0,
        frailty_shape=2.5,
        multiplicity_values=multiplicity_values,
        multiplicity_counts=counts,
        multiplicity_probabilities=probabilities,
        amount_intercept=6.0,
        amount_frailty_score=0.2,
        amount_log1p_gap_days=0.1,
        amount_household_scale=0.2,
        amount_ar1=0.3,
        amount_innovation_scale=0.15,
    )


def test_mrs2_profile_is_exact_aggregate_only_and_stop_bound() -> None:
    profile = profile_fixture_mrs2()
    payload = profile.as_dict()
    restored = MRS2Profile.from_dict(json.loads(json.dumps(payload)))

    assert restored == profile
    assert restored.sha256 == profile.sha256
    assert payload["schema_version"] == "mrs2"
    assert payload["lineage"] == {
        "mrs1_terminal_state": "STOP",
        "mrs1_stop_protocol_sha256": MRS1_STOP_PROTOCOL_SHA256,
    }
    assert payload["privacy"] == {
        "aggregate_only": True,
        "contains_ids": False,
        "contains_rows": False,
        "contains_trajectories": False,
        "contains_predictions": False,
        "contains_posterior_weights": False,
    }
    assert sum(
        value * count
        for value, count in zip(
            profile.multiplicity_values, profile.multiplicity_counts, strict=True
        )
    ) == profile.source_transaction_count
    encoded = json.dumps(payload)
    assert '"customer_id"' not in encoded and '"posterior_weights"' not in encoded


def test_exact_multi_value_multiplicity_pmf_rejects_transaction_count_drift() -> None:
    base = profile_fixture_mrs2()
    profile = replace(
        base,
        source_transaction_count=90,
        multiplicity_values=(1, 2, 5),
        multiplicity_counts=(60, 10, 2),
        multiplicity_probabilities=(60 / 72, 10 / 72, 2 / 72),
    )

    assert sum(
        value * count
        for value, count in zip(
            profile.multiplicity_values, profile.multiplicity_counts, strict=True
        )
    ) == 90
    with pytest.raises(ValueError, match="do not match transaction count"):
        replace(profile, source_transaction_count=91)


def test_profile_rejects_coercible_noncanonical_continuous_types() -> None:
    payload = profile_fixture_mrs2().as_dict()
    payload["timing"]["renewal_shape"] = "1.8"

    with pytest.raises((TypeError, ValueError), match="canonical"):
        MRS2Profile.from_dict(payload)


def test_preferred_hour_clock_has_both_exact_normalizations_and_round_trips() -> None:
    profile = profile_fixture_mrs2()
    start = datetime.fromisoformat(profile.observation_start)
    end = datetime.fromisoformat(profile.observation_end)
    base = CalendarHazardClock(profile, start, end)
    clocks = [
        CalendarHazardClock(profile, start, end, preferred_hour=hour)
        for hour in range(24)
    ]

    for observed_hour in range(24):
        assert sum(
            g
            * preferred_hour_multiplier(
                profile.preferred_hour_probabilities,
                profile.preferred_hour_strength,
                preferred_hour,
                observed_hour,
            )
            for preferred_hour, g in enumerate(profile.preferred_hour_probabilities)
        ) == pytest.approx(1, abs=1e-14)
    for preferred_hour in range(24):
        assert sum(
            g
            * preferred_hour_multiplier(
                profile.preferred_hour_probabilities,
                profile.preferred_hour_strength,
                preferred_hour,
                hour,
            )
            for hour, g in enumerate(profile.preferred_hour_probabilities)
        ) == pytest.approx(1, abs=1e-14)
        assert clocks[preferred_hour].total == base.total

    for timestamp in (
        start,
        start + timedelta(hours=11, minutes=7),
        end - timedelta(microseconds=1),
    ):
        assert sum(
            probability * clock.to_operational(timestamp)
            for probability, clock in zip(
                profile.preferred_hour_probabilities, clocks, strict=True
            )
        ) == pytest.approx(base.to_operational(timestamp), abs=1e-12)
        for clock in (clocks[0], clocks[7], clocks[23]):
            assert abs(clock.from_operational(clock.to_operational(timestamp)) - timestamp) <= timedelta(
                microseconds=1
            )

    probe = start + timedelta(hours=29, minutes=13)
    epsilon = timedelta(microseconds=100)
    for preferred_hour in (0, probe.hour, 23):
        clock = clocks[preferred_hour]
        finite_difference = (
            clock.to_operational(probe + epsilon)
            - clock.to_operational(probe - epsilon)
        ) / (2 * epsilon.total_seconds() / 86_400)
        assert finite_difference == pytest.approx(clock.jacobian(probe), rel=2e-7)


def test_positive_half_even_integer_cent_rendering() -> None:
    assert positive_half_even_cents(math.log(2.5)) == 2
    assert positive_half_even_cents(math.log(3.5)) == 4
    assert positive_half_even_cents(math.log(0.4)) == 1
    with pytest.raises(ValueError, match="finite"):
        positive_half_even_cents(math.inf)


def test_generation_is_deterministic_integer_valued_novel_and_exact_multiplicity() -> None:
    profile = profile_fixture_mrs2(multiplicity_values=(5,))
    first = generate_synthetic_transactions_mrs2(profile, customers=18, seed=20260827)
    second = generate_synthetic_transactions_mrs2(profile, customers=18, seed=20260827)

    assert first.equals(second)
    assert first["customer_id"].nunique() == 18
    assert first["amount"].dtype.kind in "iu"
    assert first["amount"].ge(1).all()
    assert set(first.groupby(["customer_id", "ts"], sort=False).size()) == {5}
    customer_id, group = next(iter(first.groupby("customer_id", sort=False)))
    fingerprint = sequence_sha256(
        list(group[["ts", "amount"]].itertuples(index=False, name=None)),
        start=datetime.fromisoformat(profile.observation_start),
    )
    with pytest.raises(RuntimeError, match="novelty"):
        MarkedRenewalSimulator(
            profile,
            start=datetime.fromisoformat(profile.observation_start),
            end=datetime.fromisoformat(profile.observation_end),
            seed=20260827,
        ).generate(1, forbidden_sequence_hashes={fingerprint})
    assert customer_id.startswith("mrs2-")
