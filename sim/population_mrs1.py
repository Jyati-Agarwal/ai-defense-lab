"""Marked periodic-renewal population model (MRS1).

MRS1 is a new model family.  Customer counts and end times are consequences of
one stationary renewal trajectory; they are never sampled or repaired
independently.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Collection, Mapping, Sequence


SCHEMA_VERSION = "mrs1"
STUDENT_T_DF = 5
JEFFREYS_PSEUDOCOUNT = 0.5
MAX_CONDITIONING_DRAWS = 100_000
MAX_RENEWALS_PER_CUSTOMER = 1_000_000


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _fixed_vector(
    name: str, values: Sequence[float], length: int, *, mean_one: bool = False
) -> tuple[float, ...]:
    result = tuple(map(float, values))
    if len(result) != length or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain {length} positive finite values")
    target = float(length) if mean_one else 1.0
    if not math.isclose(sum(result), target, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{name} must {'have mean one' if mean_one else 'sum to one'}")
    return result


def _probability_vector(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    result = tuple(map(float, values))
    if len(result) != length or any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"{name} must contain {length} nonnegative finite values")
    if not math.isclose(sum(result), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return result


@dataclass(frozen=True)
class MRS1Profile:
    """Aggregate-only coefficients for the marked periodic-renewal model."""

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
    renewal_shape: float
    mean_operational_gap_days: float
    frailty_shape: float
    multiplicity_counts: tuple[int, ...]
    multiplicity_probabilities: tuple[float, ...]
    amount_intercept: float
    amount_log_frailty: float
    amount_log1p_gap_days: float
    amount_ar1: float
    amount_innovation_scale: float

    def __post_init__(self) -> None:
        if (
            type(self.source_dataset) is not str
            or type(self.source_partition) is not str
            or not self.source_dataset
            or not self.source_partition
        ):
            raise ValueError("MRS1 source identity is required")
        if not _sha256(self.source_customer_ids_sha256) or not _sha256(
            self.source_transactions_sha256
        ):
            raise ValueError("MRS1 source hashes must be lowercase SHA-256")
        if (
            type(self.source_customer_count) is not int
            or type(self.source_transaction_count) is not int
            or self.source_customer_count < 1
            or self.source_transaction_count < self.source_customer_count
        ):
            raise ValueError("MRS1 source counts are invalid")
        start, end = datetime.fromisoformat(self.observation_start), datetime.fromisoformat(
            self.observation_end
        )
        if start.tzinfo is not None or end.tzinfo is not None or start >= end:
            raise ValueError("MRS1 observation bounds must be ordered local-wall datetimes")
        for name, counts, length in (
            ("hour_counts", self.hour_counts, 24),
            ("weekday_counts", self.weekday_counts, 7),
            ("week_counts", self.week_counts, 53),
            ("multiplicity_counts", self.multiplicity_counts, 4),
        ):
            if len(counts) != length or any(type(value) is not int or value < 0 for value in counts):
                raise ValueError(f"{name} must contain {length} nonnegative integer aggregates")
        renewal_epochs = sum(self.hour_counts)
        if (
            renewal_epochs < 1
            or renewal_epochs < self.source_customer_count
            or renewal_epochs > self.source_transaction_count
            or sum(self.weekday_counts) != renewal_epochs
            or sum(self.week_counts) != renewal_epochs
            or sum(self.multiplicity_counts) != renewal_epochs
        ):
            raise ValueError("MRS1 aggregate epoch counts are inconsistent")
        _fixed_vector("hour_weights", self.hour_weights, 24, mean_one=True)
        _fixed_vector("weekday_weights", self.weekday_weights, 7, mean_one=True)
        _fixed_vector("week_weights", self.week_weights, 53, mean_one=True)
        horizon_hours = (end - start).total_seconds() / 3_600
        for name, exposures, length in (
            ("hour_exposure_hours", self.hour_exposure_hours, 24),
            ("weekday_exposure_hours", self.weekday_exposure_hours, 7),
            ("week_exposure_hours", self.week_exposure_hours, 53),
        ):
            if len(exposures) != length or any(
                not math.isfinite(value) or value < 0 for value in exposures
            ) or not math.isclose(
                sum(exposures), horizon_hours, rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError(
                    f"{name} must contain {length} nonnegative finite window exposures"
                )
        for counts, exposures, weights in (
            (self.hour_counts, self.hour_exposure_hours, self.hour_weights),
            (
                self.weekday_counts,
                self.weekday_exposure_hours,
                self.weekday_weights,
            ),
            (self.week_counts, self.week_exposure_hours, self.week_weights),
        ):
            exposed_rates = tuple(
                (count + JEFFREYS_PSEUDOCOUNT) / exposure
                for count, exposure in zip(counts, exposures, strict=True)
                if exposure > 0
            )
            exposed_mean = sum(exposed_rates) / len(exposed_rates)
            adjusted_rates = tuple(
                (count + JEFFREYS_PSEUDOCOUNT) / exposure
                if exposure > 0
                else exposed_mean
                for count, exposure in zip(counts, exposures, strict=True)
            )
            adjusted_mean = sum(adjusted_rates) / len(adjusted_rates)
            if any(
                not math.isclose(
                    weight,
                    rate / adjusted_mean,
                    abs_tol=1e-12,
                )
                for rate, weight in zip(adjusted_rates, weights, strict=True)
            ):
                raise ValueError(
                    "MRS1 calendar weights do not match exposure-adjusted aggregates"
                )
        _probability_vector(
            "multiplicity_probabilities", self.multiplicity_probabilities, 4
        )
        if any(
            not math.isclose(probability, count / renewal_epochs, abs_tol=1e-12)
            for count, probability in zip(
                self.multiplicity_counts,
                self.multiplicity_probabilities,
                strict=True,
            )
        ):
            raise ValueError("MRS1 multiplicity probabilities do not match aggregate counts")
        if not (
            0.25 < self.renewal_shape < 10
            and 0.05 < self.mean_operational_gap_days < 365
            and 0.25 < self.frailty_shape < 20
            and math.exp(-6) < self.amount_innovation_scale < math.exp(6)
        ):
            raise ValueError("MRS1 positive model parameters are invalid")
        if not (
            -20 < self.amount_intercept < 30
            and -5 < self.amount_log_frailty < 5
            and -5 < self.amount_log1p_gap_days < 5
            and -0.95 < self.amount_ar1 < 0.95
        ):
            raise ValueError("MRS1 amount coefficients are invalid")

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_family": "marked_periodic_stationary_gamma_renewal",
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
            "timing": {
                "renewal_shape": self.renewal_shape,
                "mean_operational_gap_days": self.mean_operational_gap_days,
                "frailty_shape": self.frailty_shape,
                "stationary_first_arrival": True,
                "right_censored": True,
                "conditioned_on_at_least_one": True,
            },
            "timestamp_multiplicity": {
                "categories": ["1", "2", "3", "4+"],
                "rendered_values": [1, 2, 3, 4],
                "counts": list(self.multiplicity_counts),
                "probabilities": list(self.multiplicity_probabilities),
            },
            "amount_marks": {
                "distribution": "student_t",
                "degrees_of_freedom": STUDENT_T_DF,
                "frailty_covariate_fit": (
                    "two-stage posterior E[log U | timing] score approximation"
                ),
                "joint_shared_latent_mle": False,
                "intercept": self.amount_intercept,
                "log_frailty": self.amount_log_frailty,
                "log1p_gap_days": self.amount_log1p_gap_days,
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
    def from_dict(cls, value: Mapping[str, Any]) -> "MRS1Profile":
        if set(value) != {
            "schema_version",
            "model_family",
            "source",
            "calendar_hazard",
            "timing",
            "timestamp_multiplicity",
            "amount_marks",
            "privacy",
        } or value.get("schema_version") != SCHEMA_VERSION or value.get("model_family") != (
            "marked_periodic_stationary_gamma_renewal"
        ):
            raise ValueError("only the exact MRS1 profile schema is accepted")
        source = value["source"]
        calendar = value["calendar_hazard"]
        timing = value["timing"]
        multiplicity = value["timestamp_multiplicity"]
        amount = value["amount_marks"]
        if set(source) != {
            "dataset", "partition", "customer_ids_sha256", "transactions_sha256",
            "customers", "transactions", "observation_start", "observation_end",
        } or set(calendar) != {
            "pseudocount", "hour_counts", "weekday_counts", "week_counts",
            "hour_exposure_hours", "weekday_exposure_hours", "week_exposure_hours",
            "hour_weights", "weekday_weights", "week_weights", "normalization", "fit",
        } or set(timing) != {
            "renewal_shape", "mean_operational_gap_days", "frailty_shape",
            "stationary_first_arrival", "right_censored", "conditioned_on_at_least_one",
        } or set(multiplicity) != {
            "categories", "rendered_values", "counts", "probabilities",
        } or set(amount) != {
            "distribution", "degrees_of_freedom", "frailty_covariate_fit",
            "joint_shared_latent_mle", "intercept", "log_frailty", "log1p_gap_days",
            "ar1", "innovation_scale",
        }:
            raise ValueError("MRS1 profile structure changed")
        if calendar["pseudocount"] != JEFFREYS_PSEUDOCOUNT or calendar["normalization"] != (
            "arithmetic mean one per factor"
        ) or calendar["fit"] != (
            "Jeffreys exposure-adjusted marginal rates; unexposed bins neutral; "
            "not joint Poisson MLE"
        ) or timing["stationary_first_arrival"] is not True or timing["right_censored"] is not True or (
            timing["conditioned_on_at_least_one"] is not True
        ) or multiplicity["categories"] != ["1", "2", "3", "4+"] or multiplicity[
            "rendered_values"
        ] != [1, 2, 3, 4] or amount["distribution"] != "student_t" or amount[
            "degrees_of_freedom"
        ] != STUDENT_T_DF or amount["frailty_covariate_fit"] != (
            "two-stage posterior E[log U | timing] score approximation"
        ) or amount["joint_shared_latent_mle"] is not False or value["privacy"] != {
            "aggregate_only": True,
            "contains_ids": False,
            "contains_rows": False,
            "contains_trajectories": False,
            "contains_predictions": False,
            "contains_posterior_weights": False,
        }:
            raise ValueError("MRS1 frozen structural contract changed")
        profile = cls(
            source_dataset=str(source["dataset"]),
            source_partition=str(source["partition"]),
            source_customer_ids_sha256=str(source["customer_ids_sha256"]),
            source_transactions_sha256=str(source["transactions_sha256"]),
            source_customer_count=int(source["customers"]),
            source_transaction_count=int(source["transactions"]),
            observation_start=str(source["observation_start"]),
            observation_end=str(source["observation_end"]),
            hour_counts=tuple(map(int, calendar["hour_counts"])),
            weekday_counts=tuple(map(int, calendar["weekday_counts"])),
            week_counts=tuple(map(int, calendar["week_counts"])),
            hour_exposure_hours=tuple(map(float, calendar["hour_exposure_hours"])),
            weekday_exposure_hours=tuple(
                map(float, calendar["weekday_exposure_hours"])
            ),
            week_exposure_hours=tuple(map(float, calendar["week_exposure_hours"])),
            hour_weights=tuple(map(float, calendar["hour_weights"])),
            weekday_weights=tuple(map(float, calendar["weekday_weights"])),
            week_weights=tuple(map(float, calendar["week_weights"])),
            renewal_shape=float(timing["renewal_shape"]),
            mean_operational_gap_days=float(timing["mean_operational_gap_days"]),
            frailty_shape=float(timing["frailty_shape"]),
            multiplicity_counts=tuple(map(int, multiplicity["counts"])),
            multiplicity_probabilities=tuple(map(float, multiplicity["probabilities"])),
            amount_intercept=float(amount["intercept"]),
            amount_log_frailty=float(amount["log_frailty"]),
            amount_log1p_gap_days=float(amount["log1p_gap_days"]),
            amount_ar1=float(amount["ar1"]),
            amount_innovation_scale=float(amount["innovation_scale"]),
        )
        expected = profile.as_dict()

        def exact_json_equal(left: Any, right: Any) -> bool:
            if type(left) is not type(right):
                return False
            if isinstance(left, dict):
                return set(left) == set(right) and all(
                    exact_json_equal(left[key], right[key]) for key in left
                )
            if isinstance(left, list):
                return len(left) == len(right) and all(
                    exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
                )
            return bool(left == right)

        if not exact_json_equal(expected, value):
            raise ValueError("MRS1 profile values must use their exact canonical JSON types")
        return profile


class CalendarHazardClock:
    """Exact inversion of the profile's piecewise-constant hourly hazard."""

    def __init__(self, profile: MRS1Profile, start: datetime, end: datetime) -> None:
        if start.tzinfo is not None or end.tzinfo is not None or start >= end:
            raise ValueError("calendar bounds must be ordered local-wall datetimes")
        self.start, self.end = start, end
        self._starts: list[datetime] = []
        self._ends: list[datetime] = []
        self._weights: list[float] = []
        self._operational_starts: list[float] = []
        self._operational_ends: list[float] = []
        current, total = start, 0.0
        while current < end:
            boundary = current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            boundary = min(boundary, end)
            iso_week = current.isocalendar().week
            weight = (
                profile.hour_weights[current.hour]
                * profile.weekday_weights[current.weekday()]
                * profile.week_weights[iso_week - 1]
            )
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("calendar hazard must remain positive and finite")
            self._starts.append(current)
            self._ends.append(boundary)
            self._weights.append(weight)
            self._operational_starts.append(total)
            total += (boundary - current).total_seconds() * weight / 86_400
            self._operational_ends.append(total)
            current = boundary
        self.total = total

    def to_operational(self, timestamp: datetime) -> float:
        if timestamp.tzinfo is not None or not self.start <= timestamp <= self.end:
            raise ValueError("timestamp is outside the local-wall calendar clock")
        if timestamp == self.end:
            return self.total
        index = bisect.bisect_right(self._starts, timestamp) - 1
        return self._operational_starts[index] + (
            (timestamp - self._starts[index]).total_seconds()
            * self._weights[index]
            / 86_400
        )

    def from_operational(self, value: float) -> datetime:
        if not math.isfinite(value) or not 0 <= value < self.total:
            raise ValueError("operational time is outside the right-censored window")
        index = bisect.bisect_right(self._operational_ends, value)
        timestamp = self._starts[index] + timedelta(
            days=(value - self._operational_starts[index]) / self._weights[index]
        )
        if not self.start <= timestamp < self.end:
            raise RuntimeError("calendar-hazard inversion left its window")
        return timestamp


def sequence_sha256(rows: Sequence[tuple[datetime, float]], *, start: datetime) -> str:
    """Fingerprint one exact trajectory without retaining it in an artifact."""
    canonical = [
        [(timestamp - start).total_seconds(), float.hex(amount)]
        for timestamp, amount in rows
    ]
    return _canonical_hash({"sequence": canonical})


class MarkedRenewalSimulator:
    def __init__(
        self,
        profile: MRS1Profile,
        *,
        start: datetime,
        end: datetime,
        seed: int,
    ) -> None:
        import numpy as np

        self.profile = profile
        self.clock = CalendarHazardClock(profile, start, end)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

    def _conditioned_epochs(self) -> tuple[float, list[float]]:
        profile, rng, horizon = self.profile, self.rng, self.clock.total
        for _ in range(MAX_CONDITIONING_DRAWS):
            frailty = float(rng.gamma(profile.frailty_shape, 1 / profile.frailty_shape))
            scale = profile.mean_operational_gap_days / (profile.renewal_shape * frailty)
            containing_gap = float(rng.gamma(profile.renewal_shape + 1, scale))
            first = float(rng.random()) * containing_gap
            if first >= horizon:
                continue
            epochs = [first]
            while len(epochs) < MAX_RENEWALS_PER_CUSTOMER:
                candidate = epochs[-1] + float(rng.gamma(profile.renewal_shape, scale))
                if candidate >= horizon:
                    return frailty, epochs
                epochs.append(candidate)
            raise RuntimeError("MRS1 renewal safety ceiling exceeded; no trajectory was clipped")
        raise RuntimeError("MRS1 conditioning failed; no trajectory was substituted")

    def _trajectory(self, customer_id: str) -> list[dict[str, Any]]:
        import numpy as np

        profile, rng = self.profile, self.rng
        frailty, epochs = self._conditioned_epochs()
        rows: list[dict[str, Any]] = []
        previous_log_amount: float | None = None
        previous_baseline: float | None = None
        previous_timestamp: datetime | None = None
        for epoch in epochs:
            timestamp = self.clock.from_operational(epoch)
            multiplicity = int(rng.choice((1, 2, 3, 4), p=profile.multiplicity_probabilities))
            for _ in range(multiplicity):
                gap_days = (
                    timestamp - (previous_timestamp or self.clock.start)
                ).total_seconds() / 86_400
                baseline = (
                    profile.amount_intercept
                    + profile.amount_log_frailty * math.log(frailty)
                    + profile.amount_log1p_gap_days * math.log1p(gap_days)
                )
                conditional = baseline
                if previous_log_amount is not None and previous_baseline is not None:
                    conditional += profile.amount_ar1 * (previous_log_amount - previous_baseline)
                log_amount = conditional + profile.amount_innovation_scale * float(
                    rng.standard_t(STUDENT_T_DF)
                )
                try:
                    amount = math.exp(log_amount)
                except OverflowError as exc:
                    raise RuntimeError("MRS1 amount overflowed; no clipping was applied") from exc
                if not math.isfinite(amount) or amount <= 0:
                    raise RuntimeError("MRS1 produced a nonpositive/nonfinite amount")
                rows.append({"customer_id": customer_id, "ts": timestamp, "amount": amount})
                previous_log_amount, previous_baseline, previous_timestamp = (
                    log_amount,
                    baseline,
                    timestamp,
                )
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
        forbidden_ids, forbidden_hashes = set(forbidden_customer_ids), set(forbidden_sequence_hashes)
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index in range(customers):
            customer_id = f"mrs1-{self.seed:016x}-{index:09d}"
            if customer_id in forbidden_ids or customer_id in seen_ids:
                raise RuntimeError("MRS1 generated a non-new customer ID")
            trajectory = self._trajectory(customer_id)
            fingerprint = sequence_sha256(
                [(row["ts"], row["amount"]) for row in trajectory], start=self.clock.start
            )
            if fingerprint in forbidden_hashes:
                raise RuntimeError("MRS1 exact source-sequence novelty check failed")
            rows.extend(trajectory)
            seen_ids.add(customer_id)
        frame = pd.DataFrame(rows, columns=["customer_id", "ts", "amount"])
        if frame["customer_id"].nunique() != customers or len(seen_ids) != customers:
            raise RuntimeError("MRS1 did not realize the exact requested customer count")
        return frame.sort_values(["customer_id", "ts"], kind="stable").reset_index(drop=True)


def generate_synthetic_transactions_mrs1(
    profile: MRS1Profile,
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
