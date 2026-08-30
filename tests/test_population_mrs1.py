from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import pytest

from sim.population_mrs1 import (
    CalendarHazardClock,
    MRS1Profile,
    MarkedRenewalSimulator,
    generate_synthetic_transactions_mrs1,
    sequence_sha256,
)


def profile_fixture(*, multiplicity: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)) -> MRS1Profile:
    multiplicity_counts = tuple(int(8_904 * value) for value in multiplicity)
    assert sum(multiplicity_counts) == 8_904
    return MRS1Profile(
        source_dataset="synthetic-unit-fixture",
        source_partition="development",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=20,
        source_transaction_count=9_000,
        observation_start="2025-01-01T00:00:00",
        observation_end="2025-01-15T00:00:00",
        hour_counts=(371,) * 24,
        weekday_counts=(1_272,) * 7,
        week_counts=(168,) * 53,
        hour_exposure_hours=(14.0,) * 24,
        weekday_exposure_hours=(48.0,) * 7,
        week_exposure_hours=(336.0 / 53,) * 53,
        hour_weights=(1.0,) * 24,
        weekday_weights=(1.0,) * 7,
        week_weights=(1.0,) * 53,
        renewal_shape=1.8,
        mean_operational_gap_days=2.0,
        frailty_shape=2.5,
        multiplicity_counts=multiplicity_counts,
        multiplicity_probabilities=multiplicity,
        amount_intercept=3.0,
        amount_log_frailty=0.2,
        amount_log1p_gap_days=0.1,
        amount_ar1=0.3,
        amount_innovation_scale=0.15,
    )


def test_mrs1_profile_is_exact_aggregate_only_round_trip() -> None:
    profile = profile_fixture()
    payload = profile.as_dict()
    restored = MRS1Profile.from_dict(json.loads(json.dumps(payload)))

    assert restored == profile
    assert restored.sha256 == profile.sha256
    assert payload["schema_version"] == "mrs1"
    assert payload["privacy"] == {
        "aggregate_only": True,
        "contains_ids": False,
        "contains_rows": False,
        "contains_trajectories": False,
        "contains_predictions": False,
        "contains_posterior_weights": False,
    }
    encoded = json.dumps(payload)
    assert '"posterior_weights":' not in encoded


@pytest.mark.parametrize(
    ("section", "field", "hostile"),
    (
        ("source", "customers", True),
        ("source", "transactions", "9000"),
        ("timing", "renewal_shape", "1.8"),
        ("calendar_hazard", "hour_counts", ["371"] * 24),
    ),
)
def test_mrs1_profile_rejects_coercible_noncanonical_types(
    section: str, field: str, hostile: object
) -> None:
    payload = profile_fixture().as_dict()
    payload[section][field] = hostile

    with pytest.raises((TypeError, ValueError), match="canonical JSON types"):
        MRS1Profile.from_dict(payload)


def test_calendar_hazard_inversion_is_direct_and_round_trips() -> None:
    profile = profile_fixture()
    start = datetime(2025, 1, 1, 0, 17, 13)
    end = start + timedelta(days=2, hours=3, minutes=11)
    clock = CalendarHazardClock(profile, start, end)

    assert clock.total == pytest.approx((end - start).total_seconds() / 86_400)
    for timestamp in (
        start,
        start + timedelta(seconds=1),
        start + timedelta(hours=11, minutes=7),
        end - timedelta(microseconds=1),
    ):
        assert abs(
            clock.from_operational(clock.to_operational(timestamp)) - timestamp
        ) <= timedelta(microseconds=1)
    with pytest.raises(ValueError, match="right-censored"):
        clock.from_operational(clock.total)


def test_generation_is_deterministic_conditioned_ordered_and_novel() -> None:
    profile = profile_fixture()
    first = generate_synthetic_transactions_mrs1(profile, customers=40, seed=20260827)
    second = generate_synthetic_transactions_mrs1(profile, customers=40, seed=20260827)

    assert first.equals(second)
    assert first["customer_id"].nunique() == 40
    assert first.groupby("customer_id").size().ge(1).all()
    assert first["amount"].map(lambda value: math.isfinite(value) and value > 0).all()
    assert first["ts"].between(
        datetime.fromisoformat(profile.observation_start),
        datetime.fromisoformat(profile.observation_end),
        inclusive="left",
    ).all()
    assert all(
        group["ts"].is_monotonic_increasing
        for _, group in first.groupby("customer_id", sort=False)
    )

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
    assert customer_id.startswith("mrs1-")


def test_multiplicity_four_plus_renders_exactly_four_without_count_repair() -> None:
    profile = profile_fixture(multiplicity=(0.0, 0.0, 0.0, 1.0))
    frame = generate_synthetic_transactions_mrs1(profile, customers=12, seed=20260829)

    multiplicities = frame.groupby(["customer_id", "ts"], sort=False).size()
    assert set(multiplicities) == {4}
    assert frame.groupby("customer_id").size().map(lambda value: value % 4 == 0).all()
