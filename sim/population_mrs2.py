"""MRS2 marked renewal population model.

MRS2 keeps MRS1's stationary, right-censored renewal process and adds one
latent preferred-hour class plus finite-moment Gaussian log-cent marks.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from typing import Any, Collection, Mapping, Sequence

from sim.population_mrs1 import (
    JEFFREYS_PSEUDOCOUNT,
    MAX_CONDITIONING_DRAWS,
    MAX_RENEWALS_PER_CUSTOMER,
    MRS1Profile,
    _canonical_hash,
    _probability_vector,
    _sha256,
)


SCHEMA_VERSION = "mrs2"
MODEL_FAMILY = "preferred_hour_marked_periodic_stationary_gamma_renewal"
MRS1_STOP_PROTOCOL_SHA256 = (
    "3670720a2fd0e61b2769173c19e0d4cadd631586e0d1b958dfee9d132ea382ff"
)


def _base_hour_masses(
    *,
    start: datetime,
    end: datetime,
    hour_weights: Sequence[float],
    weekday_weights: Sequence[float],
    week_weights: Sequence[float],
) -> tuple[float, ...]:
    """Return exact operational exposure by local hour for the given window."""
    masses = [0.0] * 24
    current = start
    while current < end:
        boundary = min(
            current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            end,
        )
        weight = (
            hour_weights[current.hour]
            * weekday_weights[current.weekday()]
            * week_weights[current.isocalendar().week - 1]
        )
        masses[current.hour] += (boundary - current).total_seconds() * weight / 86_400
        current = boundary
    return tuple(masses)


def pooled_hour_probabilities(
    *,
    start: datetime,
    end: datetime,
    hour_weights: Sequence[float],
    weekday_weights: Sequence[float],
    week_weights: Sequence[float],
) -> tuple[float, ...]:
    """Base-clock operational-mass shares used for both H and latent C."""
    masses = _base_hour_masses(
        start=start,
        end=end,
        hour_weights=hour_weights,
        weekday_weights=weekday_weights,
        week_weights=week_weights,
    )
    total = sum(masses)
    if not math.isfinite(total) or total <= 0 or any(value <= 0 for value in masses):
        raise ValueError("every local hour needs positive base-clock operational exposure")
    probabilities = tuple(value / total for value in masses)
    return probabilities[:-1] + (1.0 - sum(probabilities[:-1]),)


def preferred_hour_multiplier(
    preferred_hour_probabilities: Sequence[float],
    preference_strength: float,
    preferred_hour: int,
    observed_hour: int,
) -> float:
    """r_c(h) with exact pooled and base-clock time normalizations."""
    if type(preferred_hour) is not int or not 0 <= preferred_hour < 24:
        raise ValueError("preferred_hour must be in 0..23")
    if type(observed_hour) is not int or not 0 <= observed_hour < 24:
        raise ValueError("observed_hour must be in 0..23")
    if not 0 <= preference_strength < 1:
        raise ValueError("preference_strength must be in [0, 1)")
    g = preferred_hour_probabilities[preferred_hour]
    return (1.0 - preference_strength) + (
        preference_strength / g if observed_hour == preferred_hour else 0.0
    )


def frailty_score(frailty: Any, frailty_shape: float) -> Any:
    """Bounded, mean-zero, variance-one score for U~Gamma(a, rate=a)."""
    import numpy as np
    from scipy.special import gammainc

    values = np.asarray(frailty, dtype=float)
    if not math.isfinite(float(frailty_shape)) or frailty_shape <= 0:
        raise ValueError("frailty_shape must be positive and finite")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("frailty must be positive and finite")
    result = math.sqrt(3.0) * (2.0 * gammainc(frailty_shape, frailty_shape * values) - 1.0)
    return float(result) if values.ndim == 0 else result


def positive_half_even_cents(log_cents: float) -> int:
    """Render a finite log-cent mark to a positive integer using half-even."""
    if not math.isfinite(log_cents):
        raise ValueError("log_cents must be finite")
    try:
        raw_cents = Decimal(str(math.exp(log_cents)))
    except OverflowError:
        with localcontext() as context:
            context.prec = 34
            raw_cents = Decimal(str(log_cents)).exp()
    cents = raw_cents.to_integral_value(rounding=ROUND_HALF_EVEN)
    return max(1, int(cents))


@dataclass(frozen=True)
class MRS2Profile:
    """Aggregate-only MRS2 coefficients; no household latent state is retained."""

    source_dataset: str
    source_partition: str
    source_customer_ids_sha256: str
    source_transactions_sha256: str
    source_customer_count: int
    source_transaction_count: int
    observation_start: str
    observation_end: str
    hour_counts: tuple[int, ...]
    weekday_counts: tuple[int, ...]
    week_counts: tuple[int, ...]
    hour_exposure_hours: tuple[float, ...]
    weekday_exposure_hours: tuple[float, ...]
    week_exposure_hours: tuple[float, ...]
    hour_weights: tuple[float, ...]
    weekday_weights: tuple[float, ...]
    week_weights: tuple[float, ...]
    preferred_hour_probabilities: tuple[float, ...]
    preferred_hour_strength: float
    renewal_shape: float
    mean_operational_gap_days: float
    frailty_shape: float
    multiplicity_values: tuple[int, ...]
    multiplicity_counts: tuple[int, ...]
    multiplicity_probabilities: tuple[float, ...]
    amount_intercept: float
    amount_frailty_score: float
    amount_log1p_gap_days: float
    amount_household_scale: float
    amount_ar1: float
    amount_innovation_scale: float

    def __post_init__(self) -> None:
        if (
            type(self.source_dataset) is not str
            or type(self.source_partition) is not str
            or not self.source_dataset
            or not self.source_partition
        ):
            raise ValueError("MRS2 source identity is required")
        if not _sha256(self.source_customer_ids_sha256) or not _sha256(
            self.source_transactions_sha256
        ):
            raise ValueError("MRS2 source hashes must be lowercase SHA-256")
        if (
            type(self.source_customer_count) is not int
            or type(self.source_transaction_count) is not int
            or self.source_customer_count < 1
            or self.source_transaction_count < self.source_customer_count
        ):
            raise ValueError("MRS2 source counts are invalid")
        start = datetime.fromisoformat(self.observation_start)
        end = datetime.fromisoformat(self.observation_end)
        if start.tzinfo is not None or end.tzinfo is not None or start >= end:
            raise ValueError("MRS2 observation bounds must be ordered local-wall datetimes")
        float_vectors = (
            self.hour_exposure_hours,
            self.weekday_exposure_hours,
            self.week_exposure_hours,
            self.hour_weights,
            self.weekday_weights,
            self.week_weights,
            self.preferred_hour_probabilities,
            self.multiplicity_probabilities,
        )
        if any(
            type(value) is not float
            for values in float_vectors
            for value in values
        ):
            raise ValueError("MRS2 continuous vectors must contain canonical floats")
        float_fields = (
            self.preferred_hour_strength,
            self.renewal_shape,
            self.mean_operational_gap_days,
            self.frailty_shape,
            self.amount_intercept,
            self.amount_frailty_score,
            self.amount_log1p_gap_days,
            self.amount_household_scale,
            self.amount_ar1,
            self.amount_innovation_scale,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in float_fields):
            raise ValueError("MRS2 continuous parameters must be canonical finite floats")

        if (
            not self.multiplicity_values
            or len(self.multiplicity_values) != len(self.multiplicity_counts)
            or len(self.multiplicity_values) != len(self.multiplicity_probabilities)
            or any(type(value) is not int or value < 1 for value in self.multiplicity_values)
            or tuple(sorted(set(self.multiplicity_values))) != self.multiplicity_values
            or any(type(value) is not int or value < 1 for value in self.multiplicity_counts)
        ):
            raise ValueError("MRS2 multiplicities must be exact observed positive integers")
        epochs = sum(self.multiplicity_counts)
        if sum(
            value * count
            for value, count in zip(
                self.multiplicity_values, self.multiplicity_counts, strict=True
            )
        ) != self.source_transaction_count:
            raise ValueError("MRS2 exact multiplicity counts do not match transaction count")
        # Reuse the frozen MRS1 validation for the unchanged calendar and renewal fields.
        collapsed = tuple(
            sum(
                count
                for value, count in zip(
                    self.multiplicity_values, self.multiplicity_counts, strict=True
                )
                if (value if value < 4 else 4) == category
            )
            for category in range(1, 5)
        )
        MRS1Profile(
            source_dataset=self.source_dataset,
            source_partition=self.source_partition,
            source_customer_ids_sha256=self.source_customer_ids_sha256,
            source_transactions_sha256=self.source_transactions_sha256,
            source_customer_count=self.source_customer_count,
            source_transaction_count=self.source_transaction_count,
            observation_start=self.observation_start,
            observation_end=self.observation_end,
            hour_counts=self.hour_counts,
            weekday_counts=self.weekday_counts,
            week_counts=self.week_counts,
            hour_exposure_hours=self.hour_exposure_hours,
            weekday_exposure_hours=self.weekday_exposure_hours,
            week_exposure_hours=self.week_exposure_hours,
            hour_weights=self.hour_weights,
            weekday_weights=self.weekday_weights,
            week_weights=self.week_weights,
            renewal_shape=self.renewal_shape,
            mean_operational_gap_days=self.mean_operational_gap_days,
            frailty_shape=self.frailty_shape,
            multiplicity_counts=collapsed,
            multiplicity_probabilities=tuple(value / epochs for value in collapsed),
            amount_intercept=0.0,
            amount_log_frailty=0.0,
            amount_log1p_gap_days=0.0,
            amount_ar1=0.0,
            amount_innovation_scale=1.0,
        )
        _probability_vector(
            "preferred_hour_probabilities", self.preferred_hour_probabilities, 24
        )
        _probability_vector(
            "multiplicity_probabilities",
            self.multiplicity_probabilities,
            len(self.multiplicity_values),
        )
        if any(value <= 0 for value in self.preferred_hour_probabilities):
            raise ValueError("MRS2 pooled preferred-hour probabilities must be positive")
        if any(
            not math.isclose(probability, count / epochs, rel_tol=0, abs_tol=1e-12)
            for count, probability in zip(
                self.multiplicity_counts,
                self.multiplicity_probabilities,
                strict=True,
            )
        ):
            raise ValueError("MRS2 multiplicity PMF must match exact aggregate counts")
        expected_g = pooled_hour_probabilities(
            start=start,
            end=end,
            hour_weights=self.hour_weights,
            weekday_weights=self.weekday_weights,
            week_weights=self.week_weights,
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
            for actual, expected in zip(
                self.preferred_hour_probabilities, expected_g, strict=True
            )
        ):
            raise ValueError(
                "MRS2 preferred-hour probabilities must equal base-clock operational-mass shares"
            )
        if not 0 <= self.preferred_hour_strength < 0.99:
            raise ValueError("MRS2 preferred-hour strength must be in [0, 0.99)")
        if not (
            -20 < self.amount_intercept < 30
            and -5 < self.amount_frailty_score < 5
            and -5 < self.amount_log1p_gap_days < 5
            and 0 < self.amount_household_scale < math.exp(6)
            and -0.95 < self.amount_ar1 < 0.95
            and math.exp(-6) < self.amount_innovation_scale < math.exp(6)
        ):
            raise ValueError("MRS2 amount parameters are invalid")

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            "lineage": {
                "mrs1_terminal_state": "STOP",
                "mrs1_stop_protocol_sha256": MRS1_STOP_PROTOCOL_SHA256,
            },
            "source": {
                "dataset": self.source_dataset,
                "partition": self.source_partition,
                "customer_ids_sha256": self.source_customer_ids_sha256,
                "transactions_sha256": self.source_transactions_sha256,
                "customers": self.source_customer_count,
                "transactions": self.source_transaction_count,
                "observation_start": self.observation_start,
                "observation_end": self.observation_end,
            },
            "calendar_hazard": {
                "pseudocount": JEFFREYS_PSEUDOCOUNT,
                "hour_counts": list(self.hour_counts),
                "weekday_counts": list(self.weekday_counts),
                "week_counts": list(self.week_counts),
                "hour_exposure_hours": list(self.hour_exposure_hours),
                "weekday_exposure_hours": list(self.weekday_exposure_hours),
                "week_exposure_hours": list(self.week_exposure_hours),
                "hour_weights": list(self.hour_weights),
                "weekday_weights": list(self.weekday_weights),
                "week_weights": list(self.week_weights),
                "normalization": "arithmetic mean one per factor",
                "fit": (
                    "Jeffreys exposure-adjusted marginal rates; unexposed bins neutral; "
                    "not joint Poisson MLE"
                ),
            },
            "preferred_hour": {
                "classes": list(range(24)),
                "pooled_probabilities": list(self.preferred_hour_probabilities),
                "pooled_probability_basis": "base-clock operational-mass share over exact window",
                "strength": self.preferred_hour_strength,
                "multiplier": "(1-lambda)+lambda*1[h=c]/g_c",
                "pooled_rate_normalized": True,
                "class_time_normalized": True,
            },
            "timing": {
                "renewal_shape": self.renewal_shape,
                "mean_operational_gap_days": self.mean_operational_gap_days,
                "frailty_shape": self.frailty_shape,
                "stationary_first_arrival": True,
                "right_censored": True,
                "conditioned_on_at_least_one": True,
                "latent_class_frailty_mixture": "full",
                "calendar_jacobian_in_likelihood": True,
            },
            "timestamp_multiplicity": {
                "values": list(self.multiplicity_values),
                "counts": list(self.multiplicity_counts),
                "probabilities": list(self.multiplicity_probabilities),
                "fit": "exact observed integer multiplicity PMF",
            },
            "amount_marks": {
                "unit": "integer_cents",
                "latent_scale": "log_cents",
                "distribution": "stationary_gaussian_ar1",
                "frailty_score": "sqrt(3)*(2*GammaCDF(U;a,rate=a)-1)",
                "timing_latents": "full posterior marginal over C,U",
                "household_intercept": "B~Normal(0,tau^2), analytically integrated",
                "rounding": "positive half-even integer cents",
                "intercept": self.amount_intercept,
                "frailty_score_coefficient": self.amount_frailty_score,
                "log1p_gap_days": self.amount_log1p_gap_days,
                "household_scale": self.amount_household_scale,
                "ar1": self.amount_ar1,
                "innovation_scale": self.amount_innovation_scale,
            },
            "privacy": {
                "aggregate_only": True,
                "contains_ids": False,
                "contains_rows": False,
                "contains_trajectories": False,
                "contains_predictions": False,
                "contains_posterior_weights": False,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MRS2Profile":
        if type(value) is not dict:
            raise ValueError("only the exact MRS2 profile schema is accepted")
        sections = {
            "schema_version",
            "model_family",
            "lineage",
            "source",
            "calendar_hazard",
            "preferred_hour",
            "timing",
            "timestamp_multiplicity",
            "amount_marks",
            "privacy",
        }
        if set(value) != sections or value.get("schema_version") != SCHEMA_VERSION or value.get(
            "model_family"
        ) != MODEL_FAMILY:
            raise ValueError("only the exact MRS2 profile schema is accepted")
        source = value["source"]
        calendar = value["calendar_hazard"]
        preferred = value["preferred_hour"]
        timing = value["timing"]
        multiplicity = value["timestamp_multiplicity"]
        amount = value["amount_marks"]
        profile = cls(
            source_dataset=source["dataset"],
            source_partition=source["partition"],
            source_customer_ids_sha256=source["customer_ids_sha256"],
            source_transactions_sha256=source["transactions_sha256"],
            source_customer_count=source["customers"],
            source_transaction_count=source["transactions"],
            observation_start=source["observation_start"],
            observation_end=source["observation_end"],
            hour_counts=tuple(calendar["hour_counts"]),
            weekday_counts=tuple(calendar["weekday_counts"]),
            week_counts=tuple(calendar["week_counts"]),
            hour_exposure_hours=tuple(calendar["hour_exposure_hours"]),
            weekday_exposure_hours=tuple(calendar["weekday_exposure_hours"]),
            week_exposure_hours=tuple(calendar["week_exposure_hours"]),
            hour_weights=tuple(calendar["hour_weights"]),
            weekday_weights=tuple(calendar["weekday_weights"]),
            week_weights=tuple(calendar["week_weights"]),
            preferred_hour_probabilities=tuple(preferred["pooled_probabilities"]),
            preferred_hour_strength=preferred["strength"],
            renewal_shape=timing["renewal_shape"],
            mean_operational_gap_days=timing["mean_operational_gap_days"],
            frailty_shape=timing["frailty_shape"],
            multiplicity_values=tuple(multiplicity["values"]),
            multiplicity_counts=tuple(multiplicity["counts"]),
            multiplicity_probabilities=tuple(multiplicity["probabilities"]),
            amount_intercept=amount["intercept"],
            amount_frailty_score=amount["frailty_score_coefficient"],
            amount_log1p_gap_days=amount["log1p_gap_days"],
            amount_household_scale=amount["household_scale"],
            amount_ar1=amount["ar1"],
            amount_innovation_scale=amount["innovation_scale"],
        )

        def exact(left: Any, right: Any) -> bool:
            if type(left) is not type(right):
                return False
            if isinstance(left, dict):
                return set(left) == set(right) and all(
                    exact(left[key], right[key]) for key in left
                )
            if isinstance(left, list):
                return len(left) == len(right) and all(
                    exact(a, b) for a, b in zip(left, right, strict=True)
                )
            return bool(left == right)

        if not exact(profile.as_dict(), value):
            raise ValueError("MRS2 profile values must use exact canonical JSON types")
        return profile


class CalendarHazardClock:
    """MRS1 base clock with an optional normalized preferred-hour class."""

    def __init__(
        self,
        profile: MRS2Profile,
        start: datetime,
        end: datetime,
        *,
        preferred_hour: int | None = None,
        preference_strength: float | None = None,
    ) -> None:
        if start.tzinfo is not None or end.tzinfo is not None or start >= end:
            raise ValueError("calendar bounds must be ordered local-wall datetimes")
        if preferred_hour is not None and (
            type(preferred_hour) is not int or not 0 <= preferred_hour < 24
        ):
            raise ValueError("preferred_hour must be None or in 0..23")
        strength = (
            profile.preferred_hour_strength
            if preference_strength is None
            else float(preference_strength)
        )
        if not 0 <= strength < 1:
            raise ValueError("preference_strength must be in [0, 1)")
        self.profile = profile
        self.start, self.end = start, end
        self.preferred_hour = preferred_hour
        self.preference_strength = strength
        self._starts: list[datetime] = []
        self._ends: list[datetime] = []
        self._base_weights: list[float] = []
        self._base_starts: list[float] = []
        self._base_ends: list[float] = []
        self._hour_starts: list[tuple[float, ...]] = []
        hour_totals = [0.0] * 24
        current, total = start, 0.0
        while current < end:
            boundary = min(
                current.replace(minute=0, second=0, microsecond=0)
                + timedelta(hours=1),
                end,
            )
            weight = (
                profile.hour_weights[current.hour]
                * profile.weekday_weights[current.weekday()]
                * profile.week_weights[current.isocalendar().week - 1]
            )
            duration = (boundary - current).total_seconds() * weight / 86_400
            self._starts.append(current)
            self._ends.append(boundary)
            self._base_weights.append(weight)
            self._base_starts.append(total)
            self._hour_starts.append(tuple(hour_totals))
            total += duration
            hour_totals[current.hour] += duration
            self._base_ends.append(total)
            current = boundary
        self.base_total = total
        self.hour_totals = tuple(hour_totals)
        self._class_ends_cache: dict[tuple[int, float], list[float]] = {}
        expected = tuple(value / total for value in hour_totals)
        if any(
            not math.isclose(actual, target, rel_tol=0, abs_tol=1e-12)
            for actual, target in zip(
                profile.preferred_hour_probabilities, expected, strict=True
            )
        ):
            raise ValueError("preferred-hour probabilities do not match this clock window")
        self.total = total
        self._operational_starts = [
            self._coordinate_at_segment_start(index) for index in range(len(self._starts))
        ]
        self._operational_ends = self._operational_starts[1:] + [self.total]

    def _coordinate_at_segment_start(self, index: int) -> float:
        base = self._base_starts[index]
        if self.preferred_hour is None:
            return base
        preferred = self._hour_starts[index][self.preferred_hour]
        g = self.profile.preferred_hour_probabilities[self.preferred_hour]
        return (1 - self.preference_strength) * base + self.preference_strength * preferred / g

    def _segment_index(self, timestamp: datetime) -> int:
        if timestamp.tzinfo is not None or not self.start <= timestamp < self.end:
            raise ValueError("timestamp is outside the local-wall calendar clock")
        return bisect.bisect_right(self._starts, timestamp) - 1

    def base_jacobian(self, timestamp: datetime) -> float:
        return self._base_weights[self._segment_index(timestamp)]

    def jacobian(self, timestamp: datetime) -> float:
        index = self._segment_index(timestamp)
        multiplier = 1.0
        if self.preferred_hour is not None:
            multiplier = preferred_hour_multiplier(
                self.profile.preferred_hour_probabilities,
                self.preference_strength,
                self.preferred_hour,
                timestamp.hour,
            )
        return self._base_weights[index] * multiplier

    def hour_operational(self, timestamp: datetime, hour: int) -> float:
        if type(hour) is not int or not 0 <= hour < 24:
            raise ValueError("hour must be in 0..23")
        if timestamp == self.end:
            return self.hour_totals[hour]
        index = self._segment_index(timestamp)
        value = self._hour_starts[index][hour]
        if timestamp.hour == hour:
            value += (
                (timestamp - self._starts[index]).total_seconds()
                * self._base_weights[index]
                / 86_400
            )
        return value

    def base_operational(self, timestamp: datetime) -> float:
        if timestamp == self.end:
            return self.base_total
        index = self._segment_index(timestamp)
        return self._base_starts[index] + (
            (timestamp - self._starts[index]).total_seconds()
            * self._base_weights[index]
            / 86_400
        )

    def class_operational(
        self, timestamp: datetime, preferred_hour: int, preference_strength: float
    ) -> float:
        base = self.base_operational(timestamp)
        g = self.profile.preferred_hour_probabilities[preferred_hour]
        return (1 - preference_strength) * base + preference_strength * (
            self.hour_operational(timestamp, preferred_hour) / g
        )

    def class_jacobian(
        self, timestamp: datetime, preferred_hour: int, preference_strength: float
    ) -> float:
        return self.base_jacobian(timestamp) * preferred_hour_multiplier(
            self.profile.preferred_hour_probabilities,
            preference_strength,
            preferred_hour,
            timestamp.hour,
        )

    def from_class_operational(
        self, value: float, preferred_hour: int, preference_strength: float
    ) -> datetime:
        if not math.isfinite(value) or not 0 <= value < self.total:
            raise ValueError("operational time is outside the right-censored window")
        preferred_hour_multiplier(
            self.profile.preferred_hour_probabilities,
            preference_strength,
            preferred_hour,
            preferred_hour,
        )
        key = (preferred_hour, preference_strength)
        ends = self._class_ends_cache.get(key)
        if ends is None:
            g = self.profile.preferred_hour_probabilities[preferred_hour]
            ends = []
            for index, base in enumerate(self._base_ends):
                preferred = self._hour_starts[index][preferred_hour]
                if self._starts[index].hour == preferred_hour:
                    preferred += base - self._base_starts[index]
                ends.append(
                    (1 - preference_strength) * base
                    + preference_strength * preferred / g
                )
            ends[-1] = self.total
            self._class_ends_cache[key] = ends
        index = bisect.bisect_right(ends, value)
        start_value = 0.0 if index == 0 else ends[index - 1]
        timestamp = self._starts[index] + timedelta(
            days=(value - start_value)
            / self.class_jacobian(
                self._starts[index], preferred_hour, preference_strength
            )
        )
        if not self.start <= timestamp < self.end:
            raise RuntimeError("MRS2 class-clock inversion left its window")
        return timestamp

    def to_operational(self, timestamp: datetime) -> float:
        if timestamp == self.end:
            return self.total
        if self.preferred_hour is None:
            return self.base_operational(timestamp)
        return self.class_operational(
            timestamp, self.preferred_hour, self.preference_strength
        )

    def from_operational(self, value: float) -> datetime:
        if not math.isfinite(value) or not 0 <= value < self.total:
            raise ValueError("operational time is outside the right-censored window")
        index = bisect.bisect_right(self._operational_ends, value)
        timestamp = self._starts[index] + timedelta(
            days=(value - self._operational_starts[index]) / self.jacobian(self._starts[index])
        )
        if not self.start <= timestamp < self.end:
            raise RuntimeError("MRS2 calendar-hazard inversion left its window")
        return timestamp


def sequence_sha256(rows: Sequence[tuple[datetime, int]], *, start: datetime) -> str:
    canonical = [
        [(timestamp - start).total_seconds(), amount] for timestamp, amount in rows
    ]
    return _canonical_hash({"mrs2_integer_cent_sequence": canonical})


class MarkedRenewalSimulator:
    def __init__(
        self,
        profile: MRS2Profile,
        *,
        start: datetime,
        end: datetime,
        seed: int,
    ) -> None:
        import numpy as np

        self.profile = profile
        self.base_clock = CalendarHazardClock(profile, start, end)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

    def _conditioned_epochs(self) -> tuple[int, float, list[float]]:
        profile, rng = self.profile, self.rng
        for _ in range(MAX_CONDITIONING_DRAWS):
            preferred_hour = int(
                rng.choice(24, p=profile.preferred_hour_probabilities)
            )
            frailty = float(rng.gamma(profile.frailty_shape, 1 / profile.frailty_shape))
            scale = profile.mean_operational_gap_days / (profile.renewal_shape * frailty)
            containing_gap = float(rng.gamma(profile.renewal_shape + 1, scale))
            first = float(rng.random()) * containing_gap
            if first >= self.base_clock.total:
                continue
            epochs = [first]
            while len(epochs) < MAX_RENEWALS_PER_CUSTOMER:
                candidate = epochs[-1] + float(rng.gamma(profile.renewal_shape, scale))
                if candidate >= self.base_clock.total:
                    return preferred_hour, frailty, epochs
                epochs.append(candidate)
            raise RuntimeError("MRS2 renewal safety ceiling exceeded; no trajectory was clipped")
        raise RuntimeError("MRS2 conditioning failed; no trajectory was substituted")

    def _trajectory(self, customer_id: str) -> list[dict[str, Any]]:
        profile, rng = self.profile, self.rng
        preferred_hour, frailty, epochs = self._conditioned_epochs()
        household_intercept = float(rng.normal(0.0, profile.amount_household_scale))
        residual = float(
            rng.normal(
                0.0,
                profile.amount_innovation_scale
                / math.sqrt(1.0 - profile.amount_ar1**2),
            )
        )
        rows: list[dict[str, Any]] = []
        previous_timestamp = self.base_clock.start
        score = frailty_score(frailty, profile.frailty_shape)
        for epoch in epochs:
            timestamp = self.base_clock.from_class_operational(
                epoch, preferred_hour, profile.preferred_hour_strength
            )
            multiplicity = int(
                rng.choice(
                    profile.multiplicity_values,
                    p=profile.multiplicity_probabilities,
                )
            )
            for _ in range(multiplicity):
                gap_days = (timestamp - previous_timestamp).total_seconds() / 86_400
                mean = (
                    profile.amount_intercept
                    + profile.amount_frailty_score * score
                    + profile.amount_log1p_gap_days * math.log1p(gap_days)
                    + household_intercept
                )
                amount = positive_half_even_cents(mean + residual)
                rows.append({"customer_id": customer_id, "ts": timestamp, "amount": amount})
                residual = profile.amount_ar1 * residual + float(
                    rng.normal(0.0, profile.amount_innovation_scale)
                )
                previous_timestamp = timestamp
        return rows

    def generate(
        self,
        customers: int,
        *,
        forbidden_customer_ids: Collection[str] = (),
        forbidden_sequence_hashes: Collection[str] = (),
    ) -> Any:
        import pandas as pd

        if type(customers) is not int or customers < 1:
            raise ValueError("customers must be a positive integer")
        forbidden_ids = set(forbidden_customer_ids)
        forbidden_hashes = set(forbidden_sequence_hashes)
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index in range(customers):
            customer_id = f"mrs2-{self.seed:016x}-{index:09d}"
            if customer_id in forbidden_ids or customer_id in seen_ids:
                raise RuntimeError("MRS2 generated a non-new customer ID")
            trajectory = self._trajectory(customer_id)
            fingerprint = sequence_sha256(
                [(row["ts"], row["amount"]) for row in trajectory],
                start=self.base_clock.start,
            )
            if fingerprint in forbidden_hashes:
                raise RuntimeError("MRS2 exact source-sequence novelty check failed")
            rows.extend(trajectory)
            seen_ids.add(customer_id)
        frame = pd.DataFrame(rows, columns=["customer_id", "ts", "amount"])
        if frame["customer_id"].nunique() != customers:
            raise RuntimeError("MRS2 did not realize the exact requested customer count")
        frame["amount"] = frame["amount"].astype("int64")
        return frame.sort_values(["customer_id", "ts"], kind="stable").reset_index(drop=True)


def generate_synthetic_transactions_mrs2(
    profile: MRS2Profile,
    *,
    customers: int,
    seed: int,
    start: datetime | None = None,
    end: datetime | None = None,
    forbidden_customer_ids: Collection[str] = (),
    forbidden_sequence_hashes: Collection[str] = (),
) -> Any:
    start = start or datetime.fromisoformat(profile.observation_start)
    end = end or datetime.fromisoformat(profile.observation_end)
    return MarkedRenewalSimulator(profile, start=start, end=end, seed=seed).generate(
        customers,
        forbidden_customer_ids=forbidden_customer_ids,
        forbidden_sequence_hashes=forbidden_sequence_hashes,
    )
