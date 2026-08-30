from __future__ import annotations

import json
import math
from collections import Counter

import pytest

from sim.population_v6 import (
    QUANTILE_PROBABILITIES,
    REPEAT_PRIMITIVES,
    SINGLETON_CONTINUOUS_PRIMITIVES,
    STRATUM_CELL_COUNTS,
    CheckerboardStratumV6,
)
from sim.population_v7 import (
    BehaviorProfileV7,
    PopulationSimulatorV7,
    REPEAT_INVOICE_COUNT_MAX,
)


def _quantiles(name: str) -> tuple[float, ...]:
    if name == "invoice_count":
        return (3.0,) * 32
    if name in {"active_day_position", "active_end_fraction", "span_within_end_fraction"}:
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


def profile_fixture() -> BehaviorProfileV7:
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
    pmf = [0.0] * (REPEAT_INVOICE_COUNT_MAX + 1)
    pmf[2] = pmf[4] = 0.5
    return BehaviorProfileV7(
        source_dataset="synthetic fixture",
        source_partition="calibration",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=96,
        source_transaction_count=192,
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
        repeat_invoice_count_pmf=tuple(pmf),
    )


def test_v7_repeat_pmf_schema_tail_sum_and_rank_allocation() -> None:
    profile = profile_fixture()
    payload = profile.as_dict()
    restored = BehaviorProfileV7.from_dict(json.loads(json.dumps(payload)))
    repeats = [
        row for row in profile.sample_primitives(12, seed=20260828)
        if row["invoice_count"] >= 2
    ]

    assert payload["schema_version"] == 7
    assert len(profile.repeat_invoice_count_pmf) == 257
    assert profile.repeat_invoice_count_pmf[:2] == (0.0, 0.0)
    assert profile.repeat_invoice_count_pmf[-1] == 0.0
    assert math.isclose(sum(profile.repeat_invoice_count_pmf), 1.0)
    assert Counter(row["invoice_count"] for row in repeats) == {2.0: 4, 4.0: 4}
    assert restored == profile

    invalid = list(profile.repeat_invoice_count_pmf)
    invalid[1] = 0.1
    with pytest.raises(ValueError, match="PMF"):
        BehaviorProfileV7(**{**profile.__dict__, "repeat_invoice_count_pmf": tuple(invalid)})


def test_v7_realized_total_and_renderer_preserve_each_sampled_count() -> None:
    from sim.entity_fidelity_v7 import generate_synthetic_transactions_v7

    profile = profile_fixture()
    simulator = PopulationSimulatorV7(
        seed=20260828,
        customers=12,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile,
    )
    expected = {
        customer.customer_id: round(customer.activity_weight)
        for customer in simulator.customers.values()
    }
    realized_total = sum(expected.values())
    historical_total = round(profile.activity_mean_transactions * 12)
    frame = generate_synthetic_transactions_v7(
        profile=profile, customers=12, seed=20260828
    )

    assert realized_total == 28
    assert historical_total == 24
    assert Counter(frame["customer_id"]) == expected


def test_v7_exact_size_allocator_can_rescale_only_the_count_residual() -> None:
    simulator = PopulationSimulatorV7(
        seed=20260828,
        customers=12,
        merchants_per_category=1,
        agentic_rate=0,
        behavior_profile=profile_fixture(),
    )
    assert sum(round(customer.activity_weight) for customer in simulator.customers.values()) == 28

    simulator._prepare_activity_quotas(24)

    assert sum(customer.remaining_quota or 0 for customer in simulator.customers.values()) == 24
    assert all(
        customer.remaining_quota == 1
        for customer in simulator.customers.values()
        if round(customer.activity_weight) == 1
    )
    assert all(
        (customer.remaining_quota or 0) >= 2
        for customer in simulator.customers.values()
        if round(customer.activity_weight) >= 2
    )


def test_v7_balanced_multiday_anchors_preserve_endpoints_and_gap() -> None:
    days = PopulationSimulatorV7._session_days(object(), 0, 10, 4)

    assert days == [0, 3, 7, 10]
    assert all(right - left >= 2 for left, right in zip(days, days[1:]))
    assert PopulationSimulatorV7._session_days(object(), 5, 9, 1) == [5]
    with pytest.raises(ValueError, match="two-day feasibility"):
        PopulationSimulatorV7._session_days(object(), 0, 5, 4)
