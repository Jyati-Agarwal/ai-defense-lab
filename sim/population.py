"""Seeded, stateful synthetic payment population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Iterator

from sim.mandate_twin import KeyRegistry, build_chain


ROOT = Path(__file__).resolve().parents[1]
MARKET_PRICES = ROOT / "data" / "market_prices.csv"
EVALUATION_PREREGISTRATION = ROOT / "results" / "evaluation_v4_preregistration.json"
EVALUATION_ATTEMPT = "evaluation_attempt_v3.json"
V3_FAILURE_SHA256 = "4c312af72f48ae83bb160593df1913ea49c1ab733a6118c4b56f8586b7cb9c69"
V4_LINEAGE = {
    "prior_preregistration": {
        "path": "evaluation_v3_preregistration.json",
        "sha256": "08b774b653c8edc246443cf8ccefe4d299f07481bd3b23ce1b386befa82f2401",
        "mode": "0444",
    },
    "prior_attempt": {
        "path": "evaluation_attempt_v2.json",
        "sha256": "6e3b2e615f99ff44f99f715ac46725eff550a89cecada0ef5291c7bb5505f5d5",
        "mode": "0444",
    },
    "prior_failure_release": {
        "path": "evaluation_release.json",
        "sha256": V3_FAILURE_SHA256,
        "mode": "0444",
    },
    "prior_model_sha256": "e0add66d9ede329dca413d2ce78318f70717aada26188740498e812e15a79d28",
    "prior_dataset_sha256": "a352a8978407fa888de4af0d1e1c44942ca20397da54eb4a804448439a3a8afd",
    "disclosed_vector_ids": ["GPF-A20", "GPF-C04"],
    "disclosed_family_ids": ["synthetic_identity_bust_out", "ttl_dormancy"],
}
V4_REMEDIATION = {
    "utilisation_current_min": 0.90,
    "utilisation_delta_min": 0.70,
    "velocity_1h_min": 5,
    "threshold_changed": False,
    "detection_evaluator_changed": False,
    "evaluation_seed_changed": False,
}
V4_GENERATOR = {
    "seed": 20260829,
    "events": 50_000,
    "fraud_rate": 0.0055,
    "benign_adversarial_rate": 0.02,
    "family_definitions_reused": True,
    "fresh_rows_only": True,
}
V4_ARENA = {
    "seed": 20260827,
    "replay_challenge_count": 48,
    "attack_success_threshold": 0.60,
    "training_partitions": ["A", "B"],
    "gate": "final == 0 or final < baseline",
    "zero_residual_is_floor_pass": True,
}
V4_EVALUATION = {
    "seed": 20260827,
    "detection_threshold": 0.30,
    "training_partitions": ["A", "B"],
    "partitions_scored_once": ["C", "D", "E"],
    "shuffle_retrains": 50,
    "shuffle_permutations": 9_999,
    "latency_benchmark_scope": "A+B",
    "protected_partition_inference_passes": 1,
}
V4_CUSTODY = {
    "prior_families_disclosed": True,
    "new_rows_not_new_families": True,
    "training_input_physically_mixed": True,
    "v3_evaluation_attempt_created": False,
    "attempt_receipt": EVALUATION_ATTEMPT,
    "retry_after_consumption": False,
    "independent_holdout": False,
}
CITIES = (
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.6139, 77.2090),
    ("Bengaluru", 12.9716, 77.5946),
    ("Pune", 18.5204, 73.8567),
    ("Chennai", 13.0827, 80.2707),
    ("Hyderabad", 17.3850, 78.4867),
)
COPULA_PRIMITIVES = (
    "invoice_count",
    "active_start_fraction",
    "active_duration_fraction",
    "relative_log_amount_center",
    "eb_log_amount_scale",
    "eb_short_session_probability",
    "circadian_phase_latent",
    "eb_circadian_concentration",
)
SINGLETON_PRIMITIVES = (
    "active_start_fraction",
    "relative_log_amount_center",
    "circadian_phase_latent",
)
COPULA_QUANTILE_PROBABILITIES = tuple(
    NormalDist().cdf(-3.1 + 6.2 * index / 31) for index in range(32)
)


@dataclass
class Customer:
    customer_id: str
    income_band: str
    circadian_weights: tuple[float, ...]
    home_geo: tuple[float, float]
    device_ids: list[str]
    merchant_affinity: dict[str, int]
    avg_ticket_minor: float
    amount_log_sigma: float = 0.52
    activity_weight: float = 1.0
    active_start_fraction: float = 0.0
    active_duration_fraction: float = 1.0
    short_session_probability: float = 0.0
    fixed_hour: int | None = None
    remaining_quota: int | None = None
    txn_history: list[dict[str, Any]] = field(default_factory=list)
    known_beneficiaries: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class BehaviorProfile:
    """Fixed-size calibration aggregate; never contains customer vectors or rows."""

    source_dataset: str
    source_partition: str
    source_customer_ids_sha256: str
    source_transactions_sha256: str
    source_customer_count: int
    source_transaction_count: int
    quantile_probabilities: tuple[float, ...]
    primitive_quantiles: tuple[tuple[float, ...], ...]
    singleton_quantiles: tuple[tuple[float, ...], ...]
    copula_correlation: tuple[tuple[float, ...], ...]
    hour_weights: tuple[float, ...]
    calendar_weights: tuple[float, ...]
    observation_days: float
    session_gap_log_mean: float
    session_gap_log_std: float
    repeat_same_day_probability: float
    singleton_probability: float
    circadian_phase_cut: float
    amount_variance_prior_alpha: float
    amount_variance_prior_beta: float
    short_session_prior_alpha: float
    short_session_prior_beta: float

    def __post_init__(self) -> None:
        dimension = len(COPULA_PRIMITIVES)
        if self.source_partition != "calibration":
            raise ValueError("behavior profiles may only be fitted on calibration customers")
        if self.quantile_probabilities != COPULA_QUANTILE_PROBABILITIES:
            raise ValueError("behavior profile must use the frozen 32 quantile knots")
        if (
            len(self.primitive_quantiles) != dimension
            or any(len(values) != 32 for values in self.primitive_quantiles)
            or any(
                left > right
                for values in self.primitive_quantiles
                for left, right in zip(values, values[1:])
            )
        ):
            raise ValueError("behavior profile primitive quantiles are invalid")
        if (
            len(self.singleton_quantiles) != len(SINGLETON_PRIMITIVES)
            or any(len(values) != 32 for values in self.singleton_quantiles)
            or any(
                left > right
                for values in self.singleton_quantiles
                for left, right in zip(values, values[1:])
            )
        ):
            raise ValueError("behavior profile singleton quantiles are invalid")
        if (
            len(self.copula_correlation) != dimension
            or any(len(row) != dimension for row in self.copula_correlation)
            or any(
                not math.isclose(self.copula_correlation[index][index], 1.0, abs_tol=1e-8)
                for index in range(dimension)
            )
            or any(
                not math.isclose(
                    self.copula_correlation[left][right],
                    self.copula_correlation[right][left],
                    abs_tol=1e-8,
                )
                for left in range(dimension)
                for right in range(dimension)
            )
        ):
            raise ValueError("behavior profile copula correlation is invalid")
        if len(self.hour_weights) != 24 or any(weight < 0 for weight in self.hour_weights):
            raise ValueError("behavior profile needs 24 nonnegative hour weights")
        if not math.isclose(sum(self.hour_weights), 1.0, abs_tol=1e-9):
            raise ValueError("behavior profile hour weights must sum to one")
        if (
            len(self.calendar_weights) != 12
            or any(weight < 0 for weight in self.calendar_weights)
            or not math.isclose(sum(self.calendar_weights), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("behavior profile circadian or calendar parameters are invalid")
        if (
            self.source_customer_count < 1
            or self.source_transaction_count < self.source_customer_count
            or self.observation_days <= 0
            or not math.isfinite(self.session_gap_log_mean)
            or self.session_gap_log_std < 0
            or not 0 <= self.repeat_same_day_probability <= 1
            or not 0 <= self.singleton_probability <= 1
            or not 0 <= self.circadian_phase_cut < 2 * math.pi
            or min(
                self.amount_variance_prior_alpha,
                self.amount_variance_prior_beta,
                self.short_session_prior_alpha,
                self.short_session_prior_beta,
            )
            <= 0
        ):
            raise ValueError("behavior profile source or session parameters are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "source": {
                "dataset": self.source_dataset,
                "partition": self.source_partition,
                "customer_ids_sha256": self.source_customer_ids_sha256,
                "transactions_sha256": self.source_transactions_sha256,
                "customers": self.source_customer_count,
                "transactions": self.source_transaction_count,
            },
            "quantile_probabilities": list(self.quantile_probabilities),
            "primitive_quantiles": {
                name: list(values)
                for name, values in zip(COPULA_PRIMITIVES, self.primitive_quantiles, strict=True)
            },
            "singleton_quantiles": {
                name: list(values)
                for name, values in zip(
                    SINGLETON_PRIMITIVES, self.singleton_quantiles, strict=True
                )
            },
            "copula_correlation": [list(row) for row in self.copula_correlation],
            "hour_weights": list(self.hour_weights),
            "calendar_weights": list(self.calendar_weights),
            "observation_days": self.observation_days,
            "session_gap_log_mean": self.session_gap_log_mean,
            "session_gap_log_std": self.session_gap_log_std,
            "repeat_same_day_probability": self.repeat_same_day_probability,
            "singleton_probability": self.singleton_probability,
            "circadian_phase_cut": self.circadian_phase_cut,
            "amount_variance_prior_alpha": self.amount_variance_prior_alpha,
            "amount_variance_prior_beta": self.amount_variance_prior_beta,
            "short_session_prior_alpha": self.short_session_prior_alpha,
            "short_session_prior_beta": self.short_session_prior_beta,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorProfile":
        if value.get("schema_version") != 3:
            raise ValueError("only frozen v3 behavior profiles are accepted")
        source = value["source"]
        primitive_quantiles = value["primitive_quantiles"]
        singleton_quantiles = value["singleton_quantiles"]
        if tuple(primitive_quantiles) != COPULA_PRIMITIVES:
            raise ValueError("behavior profile primitive order changed")
        if tuple(singleton_quantiles) != SINGLETON_PRIMITIVES:
            raise ValueError("behavior profile singleton primitive order changed")
        return cls(
            source_dataset=source["dataset"],
            source_partition=source["partition"],
            source_customer_ids_sha256=source["customer_ids_sha256"],
            source_transactions_sha256=source["transactions_sha256"],
            source_customer_count=int(source["customers"]),
            source_transaction_count=int(source["transactions"]),
            quantile_probabilities=tuple(value["quantile_probabilities"]),
            primitive_quantiles=tuple(
                tuple(primitive_quantiles[name]) for name in COPULA_PRIMITIVES
            ),
            singleton_quantiles=tuple(
                tuple(singleton_quantiles[name]) for name in SINGLETON_PRIMITIVES
            ),
            copula_correlation=tuple(tuple(row) for row in value["copula_correlation"]),
            hour_weights=tuple(value["hour_weights"]),
            calendar_weights=tuple(value["calendar_weights"]),
            observation_days=float(value["observation_days"]),
            session_gap_log_mean=float(value["session_gap_log_mean"]),
            session_gap_log_std=float(value["session_gap_log_std"]),
            repeat_same_day_probability=float(value["repeat_same_day_probability"]),
            singleton_probability=float(value["singleton_probability"]),
            circadian_phase_cut=float(value["circadian_phase_cut"]),
            amount_variance_prior_alpha=float(value["amount_variance_prior_alpha"]),
            amount_variance_prior_beta=float(value["amount_variance_prior_beta"]),
            short_session_prior_alpha=float(value["short_session_prior_alpha"]),
            short_session_prior_beta=float(value["short_session_prior_beta"]),
        )

    @property
    def activity_mean_transactions(self) -> float:
        return self.source_transaction_count / self.source_customer_count

    def sample_primitives(self, count: int, *, seed: int) -> list[dict[str, float]]:
        """Draw a singleton hurdle plus novel repeat customers from the copula."""
        import numpy as np
        from scipy.special import ndtr

        correlation = np.asarray(self.copula_correlation, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        if eigenvalues.min() < -1e-8:
            raise ValueError("behavior profile copula correlation is not PSD")
        factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
        rng = np.random.default_rng(seed)
        singleton_count = round(self.singleton_probability * count)
        singleton_indices = set(
            map(int, rng.choice(count, singleton_count, replace=False))
            if singleton_count
            else ()
        )
        repeated_indices = [index for index in range(count) if index not in singleton_indices]
        normals = rng.standard_normal(
            (len(repeated_indices), len(COPULA_PRIMITIVES))
        ) @ factor.T
        uniforms = ndtr(normals)
        probabilities = np.asarray(self.quantile_probabilities)
        repeat_samples = np.column_stack(
            [
                np.interp(uniforms[:, index], probabilities, values)
                for index, values in enumerate(self.primitive_quantiles)
            ]
        )
        repeat_same_day_count = round(
            self.repeat_same_day_probability * len(repeated_indices)
        )
        repeat_same_day_rows = set(
            map(
                int,
                rng.choice(
                    len(repeated_indices), repeat_same_day_count, replace=False
                ),
            )
            if repeat_same_day_count
            else ()
        )
        output: list[dict[str, float] | None] = [None] * count
        for repeat_row, (index, values) in enumerate(
            zip(repeated_indices, repeat_samples, strict=True)
        ):
            row = dict(zip(COPULA_PRIMITIVES, map(float, values), strict=True))
            row["invoice_count"] = float(max(2, round(row["invoice_count"])))
            row["active_start_fraction"] = min(1.0, max(0.0, row["active_start_fraction"]))
            row["active_duration_fraction"] = min(
                1.0 - row["active_start_fraction"],
                max(0.0, row["active_duration_fraction"]),
            )
            if repeat_row in repeat_same_day_rows:
                row["active_duration_fraction"] = 0.0
            row["eb_log_amount_scale"] = max(1e-6, row["eb_log_amount_scale"])
            row["eb_short_session_probability"] = min(
                1.0, max(0.0, row["eb_short_session_probability"])
            )
            phase = (row["circadian_phase_latent"] + self.circadian_phase_cut) % (2 * math.pi)
            row["circadian_phase_sin"] = math.sin(phase)
            row["circadian_phase_cos"] = math.cos(phase)
            row["eb_circadian_concentration"] = max(
                0.0, row["eb_circadian_concentration"]
            )
            output[index] = row
        singleton_uniforms = rng.random((singleton_count, len(SINGLETON_PRIMITIVES)))
        singleton_samples = np.column_stack(
            [
                np.interp(singleton_uniforms[:, index], probabilities, values)
                for index, values in enumerate(self.singleton_quantiles)
            ]
        )
        supported_hours = [hour for hour, weight in enumerate(self.hour_weights) if weight > 0]
        for singleton_row, (index, values) in enumerate(
            zip(sorted(singleton_indices), singleton_samples, strict=True)
        ):
            sampled = dict(zip(SINGLETON_PRIMITIVES, map(float, values), strict=True))
            phase = (sampled["circadian_phase_latent"] + self.circadian_phase_cut) % (
                2 * math.pi
            )
            hour = min(
                supported_hours,
                key=lambda value: abs(
                    math.atan2(
                        math.sin(phase - 2 * math.pi * value / 24),
                        math.cos(phase - 2 * math.pi * value / 24),
                    )
                ),
            )
            output[index] = {
                "invoice_count": 1.0,
                "active_start_fraction": min(
                    1.0, max(0.0, sampled["active_start_fraction"])
                ),
                "active_duration_fraction": 0.0,
                "relative_log_amount_center": sampled["relative_log_amount_center"],
                "eb_log_amount_scale": 0.0,
                "eb_short_session_probability": 0.0,
                "circadian_phase_latent": sampled["circadian_phase_latent"],
                "circadian_phase_sin": math.sin(2 * math.pi * hour / 24),
                "circadian_phase_cos": math.cos(2 * math.pi * hour / 24),
                "eb_circadian_concentration": 0.0,
                "fixed_hour": float(hour),
            }
        if any(row is None for row in output):
            raise RuntimeError("behavior hurdle failed to sample every customer")
        return output  # type: ignore[return-value]

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class Merchant:
    merchant_id: str
    payee_did: str
    beneficiary_id: str
    category: str
    mcc: str
    median_ticket_minor: int
    market_median_minor: int
    p10_minor: int
    p90_minor: int
    sku_prefix: str
    item_name: str
    risk: float
    popularity: float
    geo: tuple[float, float]


@dataclass
class Device:
    device_id: str
    owner_id: str
    trust: float
    first_seen: datetime
    last_seen: datetime


@dataclass
class Beneficiary:
    beneficiary_id: str
    first_seen: datetime
    sender_ids: set[str] = field(default_factory=set)
    payment_count: int = 0

    @property
    def in_degree(self) -> int:
        return len(self.sender_ids)


def _scores() -> dict[str, None]:
    return {key: None for key in ("semantic", "provenance", "temporal", "economic", "structural", "fused")}


def _decision() -> dict[str, Any]:
    return {"action": None, "legacy_rules_action": None, "reason_codes": [], "latency_ms": None}


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class PopulationSimulator:
    """Entities persist and evolve across calls; this is not an independent row sampler."""

    def __init__(
        self,
        *,
        seed: int = 20260827,
        customers: int = 800,
        merchants_per_category: int = 6,
        agentic_rate: float = 0.30,
        calibration_pairs: list[tuple[int, int]] | None = None,
        behavior_profile: BehaviorProfile | None = None,
    ) -> None:
        if customers < 1 or merchants_per_category < 1:
            raise ValueError("population sizes must be positive")
        if not 0 <= agentic_rate <= 1:
            raise ValueError("agentic_rate must be between zero and one")
        self.seed = seed
        self.rng = random.Random(seed)
        self.agentic_rate = agentic_rate
        self.calibration_pairs = calibration_pairs
        self.behavior_profile = behavior_profile
        self.clock = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._profile_start = self.clock
        self.event_counter = 0
        self.registry = KeyRegistry()
        self.market = self._load_market()
        self.merchants = self._build_merchants(merchants_per_category)
        self.devices: dict[str, Device] = {}
        self.customers = self._build_customers(customers)
        aggregate = list(behavior_profile.hour_weights) if behavior_profile else [
            sum(customer.circadian_weights[hour] for customer in self.customers.values()) / len(self.customers)
            for hour in range(24)
        ]
        peak = max(aggregate)
        self.hour_activity = tuple(weight / peak for weight in aggregate)
        self._session_customer_id: str | None = None
        self._quota_prepared_for: int | None = None
        self._schedule: list[tuple[datetime, str]] = []
        self._schedule_index = 0
        self._scheduled_customer_id: str | None = None
        self.beneficiaries = {
            merchant.beneficiary_id: Beneficiary(merchant.beneficiary_id, self.clock)
            for merchant in self.merchants.values()
        }

    @staticmethod
    def _load_market() -> list[dict[str, str]]:
        with MARKET_PRICES.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _build_merchants(self, per_category: int) -> dict[str, Merchant]:
        merchants: dict[str, Merchant] = {}
        rank = 1
        for price in self.market:
            for offset in range(per_category):
                merchant_id = f"mer_{rank:04d}"
                _, lat, lon = CITIES[(rank + offset) % len(CITIES)]
                median = int(price["median_minor"])
                price_factor = math.exp(self.rng.normalvariate(0, 0.12))
                merchants[merchant_id] = Merchant(
                    merchant_id=merchant_id,
                    payee_did=f"did:example:merchant-{rank:04d}",
                    beneficiary_id=f"ben_{rank:04d}",
                    category=price["category"],
                    mcc=price["mcc"],
                    median_ticket_minor=max(1, round(median * price_factor)),
                    market_median_minor=median,
                    p10_minor=int(price["p10_minor"]),
                    p90_minor=int(price["p90_minor"]),
                    sku_prefix=price["sku_prefix"],
                    item_name=price["item_name"],
                    risk=self.rng.betavariate(2, 18),
                    popularity=1 / rank**1.08,
                    geo=(lat + self.rng.uniform(-0.08, 0.08), lon + self.rng.uniform(-0.08, 0.08)),
                )
                rank += 1
        return merchants

    def _build_customers(self, count: int) -> dict[str, Customer]:
        income_tickets = {"low": 120_000, "middle": 350_000, "high": 1_200_000}
        merchant_ids = list(self.merchants)
        customers: dict[str, Customer] = {}
        primitive_samples = (
            self.behavior_profile.sample_primitives(count, seed=self.seed + 101)
            if self.behavior_profile
            else None
        )
        for number in range(1, count + 1):
            customer_id = f"cus_{number:05d}"
            income = self.rng.choices(("low", "middle", "high"), (0.45, 0.42, 0.13))[0]
            _, lat, lon = self.rng.choice(CITIES)
            if self.behavior_profile:
                primitives = primitive_samples[number - 1]
                phase = math.atan2(
                    primitives["circadian_phase_sin"], primitives["circadian_phase_cos"]
                )
                concentration = min(32.0, primitives["eb_circadian_concentration"])
                raw_hour_weights = tuple(
                    weight
                    * math.exp(
                        concentration
                        * (math.cos(2 * math.pi * hour / 24 - phase) - 1)
                    )
                    for hour, weight in enumerate(self.behavior_profile.hour_weights)
                )
                total = sum(raw_hour_weights)
                weights = tuple(weight / total for weight in raw_hour_weights)
                activity_weight = primitives["invoice_count"]
            else:
                phase = self.rng.choice((-2, -1, 0, 0, 1, 2))
                weights = tuple(
                    0.08
                    + math.exp(-(((hour - ((19 + phase) % 24) + 12) % 24 - 12) ** 2) / 18)
                    + 0.45 * math.exp(-(((hour - 12 + 12) % 24 - 12) ** 2) / 10)
                    for hour in range(24)
                )
                activity_weight = 1.0
            device_ids = []
            for device_number in range(1, self.rng.choices((2, 3, 4), (0.68, 0.27, 0.05))[0]):
                device_id = f"dev_{number:05d}_{device_number}"
                device_ids.append(device_id)
                age_days = self.rng.randint(30, 900)
                first_seen = self.clock - timedelta(days=age_days)
                self.devices[device_id] = Device(
                    device_id=device_id,
                    owner_id=customer_id,
                    trust=self.rng.uniform(0.72, 0.99),
                    first_seen=first_seen,
                    last_seen=first_seen,
                )
            affinity_ids = self.rng.sample(merchant_ids, k=self.rng.randint(4, 10))
            avg_ticket_minor = income_tickets[income] * math.exp(self.rng.normalvariate(0, 0.35))
            if self.behavior_profile:
                avg_ticket_minor = 350_000 * math.exp(primitives["relative_log_amount_center"])
            customers[customer_id] = Customer(
                customer_id=customer_id,
                income_band=income,
                circadian_weights=weights,
                home_geo=(lat + self.rng.uniform(-0.12, 0.12), lon + self.rng.uniform(-0.12, 0.12)),
                device_ids=device_ids,
                merchant_affinity={merchant_id: self.rng.randint(1, 8) for merchant_id in affinity_ids},
                avg_ticket_minor=avg_ticket_minor,
                amount_log_sigma=(
                    primitives["eb_log_amount_scale"] if self.behavior_profile else 0.52
                ),
                activity_weight=activity_weight,
                active_start_fraction=(
                    primitives["active_start_fraction"] if self.behavior_profile else 0.0
                ),
                active_duration_fraction=(
                    primitives["active_duration_fraction"] if self.behavior_profile else 1.0
                ),
                short_session_probability=(
                    primitives["eb_short_session_probability"] if self.behavior_profile else 0.0
                ),
                fixed_hour=(
                    int(primitives["fixed_hour"])
                    if self.behavior_profile and "fixed_hour" in primitives
                    else None
                ),
            )
        return customers

    def _advance_clock(self, target_hour: int | None = None) -> datetime:
        if self.behavior_profile and target_hour is None:
            if self._schedule_index >= len(self._schedule):
                raise RuntimeError("profiled schedule is exhausted; prepare the exact event quota first")
            timestamp, customer_id = self._schedule[self._schedule_index]
            self._schedule_index += 1
            self._scheduled_customer_id = customer_id
            self.clock = timestamp
            return self.clock
        if target_hour is None:
            candidate = self.clock
            mean_seconds = 8.0
            calendar_activity: tuple[float, ...] | None = None
            while True:
                candidate += timedelta(seconds=max(1, round(self.rng.expovariate(1 / mean_seconds))))
                calendar_acceptance = 1.0
                if calendar_activity:
                    observation_days = (
                        self.behavior_profile.activity_mean_transactions
                        / self.behavior_profile.transactions_per_customer_day
                    )
                    progress = (candidate - self._profile_start).total_seconds() / (
                        observation_days * 86_400
                    )
                    calendar_acceptance = calendar_activity[min(11, max(0, int(progress * 12)))]
                if self.rng.random() <= self.hour_activity[candidate.hour] * calendar_acceptance:
                    self.clock = candidate
                    break
        else:
            candidate = self.clock.replace(
                hour=target_hour,
                minute=self.rng.randrange(60),
                second=self.rng.randrange(60),
            )
            if candidate <= self.clock:
                candidate += timedelta(days=1)
            self.clock = candidate
        return self.clock

    def synchronize_clock(self, event: dict[str, Any]) -> None:
        timestamp = datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00"))
        if timestamp > self.clock:
            self.clock = timestamp

    def _prepare_activity_quotas(self, events: int) -> None:
        if not self.behavior_profile or self._quota_prepared_for == events:
            return
        population = list(self.customers.values())
        if events < len(population):
            raise ValueError("profiled generation needs at least one event per configured customer")
        extras = events - len(population)
        raw = [max(0.0, customer.activity_weight - 1.0) for customer in population]
        total = sum(raw)
        scaled = [extras * weight / total for weight in raw] if total else [extras / len(raw)] * len(raw)
        quotas = [1 + math.floor(value) for value in scaled]
        remainder = events - sum(quotas)
        order = sorted(
            range(len(population)),
            key=lambda index: (scaled[index] - math.floor(scaled[index]), population[index].customer_id),
            reverse=True,
        )
        for index in order[:remainder]:
            quotas[index] += 1
        for customer, quota in zip(population, quotas, strict=True):
            customer.remaining_quota = quota
        schedules = [self._customer_schedule(customer, quota) for customer, quota in zip(population, quotas, strict=True)]
        self._schedule = list(heapq.merge(*schedules))
        if len(self._schedule) != events:
            raise RuntimeError("profiled customer schedules did not preserve the exact event quota")
        self._schedule_index = 0
        self._quota_prepared_for = events

    def _calendar_fraction(self, lower: float, upper: float) -> float:
        if upper <= lower:
            return lower
        weighted_intervals: list[tuple[float, float, float]] = []
        for index, weight in enumerate(self.behavior_profile.calendar_weights):
            start = max(lower, index / 12)
            end = min(upper, (index + 1) / 12)
            if end > start and weight > 0:
                weighted_intervals.append((start, end, weight * (end - start) * 12))
        if not weighted_intervals:
            return self.rng.uniform(lower, upper)
        start, end, _ = self.rng.choices(
            weighted_intervals,
            weights=[interval[2] for interval in weighted_intervals],
        )[0]
        return self.rng.uniform(start, end)

    def _timestamp_at(self, fraction: float, customer: Customer) -> datetime:
        day = min(
            max(0, int(fraction * self.behavior_profile.observation_days)),
            max(0, math.ceil(self.behavior_profile.observation_days) - 1),
        )
        hour = (
            customer.fixed_hour
            if customer.fixed_hour is not None
            else self.rng.choices(range(24), weights=customer.circadian_weights)[0]
        )
        return self._profile_start + timedelta(
            days=day,
            hours=hour,
            minutes=self.rng.randrange(60),
            seconds=self.rng.randrange(60),
        )

    def _customer_schedule(self, customer: Customer, quota: int) -> list[tuple[datetime, str]]:
        lower = customer.active_start_fraction
        upper = min(1.0, lower + customer.active_duration_fraction)
        session_sizes: list[int] = []
        remaining = quota
        while remaining:
            size = 1
            remaining -= 1
            while remaining and self.rng.random() < customer.short_session_probability:
                size += 1
                remaining -= 1
            session_sizes.append(size)
        if (
            quota > 1
            and len(session_sizes) == 1
            and customer.active_duration_fraction * self.behavior_profile.observation_days > 1
        ):
            session_sizes = [quota - 1, 1]
        anchors = sorted(self._calendar_fraction(lower, upper) for _ in session_sizes)
        if len(anchors) > 1:
            anchors[0], anchors[-1] = lower, upper
        timestamps: list[datetime] = []
        window_start = self._profile_start + timedelta(days=lower * self.behavior_profile.observation_days)
        window_end = self._profile_start + timedelta(days=upper * self.behavior_profile.observation_days + 1)
        for anchor, size in zip(anchors, session_sizes, strict=True):
            base = self._timestamp_at(anchor, customer)
            offsets = [0.0]
            for _ in range(size - 1):
                gap = min(
                    86_400.0,
                    max(
                        60.0,
                        self.rng.lognormvariate(
                            self.behavior_profile.session_gap_log_mean,
                            self.behavior_profile.session_gap_log_std,
                        ),
                    ),
                )
                offsets.append(offsets[-1] + gap)
            available_seconds = max(1.0, (window_end - window_start).total_seconds() - 1)
            if offsets[-1] > available_seconds:
                scale = available_seconds / offsets[-1]
                offsets = [offset * scale for offset in offsets]
            base = min(max(base, window_start), window_end - timedelta(seconds=offsets[-1] + 1))
            timestamps.extend(base + timedelta(seconds=offset) for offset in offsets)
        timestamps.sort()
        for index in range(1, len(timestamps)):
            if timestamps[index] <= timestamps[index - 1]:
                timestamps[index] = timestamps[index - 1] + timedelta(seconds=1)
        return [(timestamp, customer.customer_id) for timestamp in timestamps]

    def _choose_customer(self, hour: int) -> Customer:
        population = list(self.customers.values())
        if self.behavior_profile:
            if self._scheduled_customer_id is None:
                raise RuntimeError("profiled customer selection requires a prepared schedule")
            customer = self.customers[self._scheduled_customer_id]
            self._scheduled_customer_id = None
            if customer.remaining_quota is None or customer.remaining_quota < 1:
                raise RuntimeError("profiled customer schedule exceeded its exact quota")
            customer.remaining_quota -= 1
            return customer
        if self.behavior_profile and self._session_customer_id is not None:
            previous = self.customers[self._session_customer_id]
            if previous.remaining_quota and self.rng.random() < self.behavior_profile.session_repeat_probability:
                previous.remaining_quota -= 1
                return previous
        available = [customer for customer in population if customer.remaining_quota is None or customer.remaining_quota > 0]
        customer = self.rng.choices(
            available,
            weights=[
                customer.circadian_weights[hour]
                * (customer.remaining_quota or customer.activity_weight)
                for customer in available
            ],
        )[0]
        if customer.remaining_quota is not None:
            customer.remaining_quota -= 1
        return customer

    def _choose_merchant(self, customer: Customer, category: str | None = None) -> Merchant:
        candidates = [merchant for merchant in self.merchants.values() if category is None or merchant.category == category]
        if category is None and self.rng.random() < 0.72:
            familiar = [self.merchants[merchant_id] for merchant_id in customer.merchant_affinity]
            return self.rng.choices(
                familiar,
                weights=[customer.merchant_affinity[merchant.merchant_id] for merchant in familiar],
            )[0]
        return self.rng.choices(candidates, weights=[merchant.popularity for merchant in candidates])[0]

    def _choose_channel(self) -> tuple[str, str]:
        if self.rng.random() < self.agentic_rate:
            return "agentic_checkout", self.rng.choices(("card", "upi"), (0.82, 0.18))[0]
        channel = self.rng.choices(
            ("card_ecom", "upi_p2p", "upi_collect", "wire"),
            (0.52, 0.27, 0.11, 0.10),
        )[0]
        return channel, {"card_ecom": "card", "upi_p2p": "upi", "upi_collect": "upi", "wire": "wire"}[channel]

    def _amount_and_hour(
        self, merchant: Merchant, weird: bool, customer: Customer | None = None
    ) -> tuple[int, int | None]:
        if self.calibration_pairs:
            amount_minor, hour = self.rng.choice(self.calibration_pairs)
            return max(0, amount_minor), hour
        if weird:
            return max(merchant.p90_minor, round(merchant.median_ticket_minor * 1.8)), 2
        if self.behavior_profile and customer is not None:
            amount = round(
                math.exp(
                    self.rng.normalvariate(
                        math.log(customer.avg_ticket_minor), customer.amount_log_sigma
                    )
                )
            )
            return max(1, amount), None
        else:
            amount = round(math.exp(self.rng.normalvariate(math.log(merchant.median_ticket_minor), 0.52)))
        return max(1, min(amount, merchant.p90_minor * 2)), None

    def generate_legitimate(
        self,
        *,
        split: str = "A",
        weird: bool = False,
        force_channel: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        if split not in {"A", "E"}:
            raise ValueError("legitimate events use split A or E")
        if weird:
            split = "E"
            category = category or self.rng.choice(("electronics", "travel", "jewellery"))
        provisional_customer = self.rng.choice(list(self.customers.values()))
        merchant = self._choose_merchant(provisional_customer, category)
        amount_minor, calibrated_hour = (
            (1, None)
            if self.behavior_profile
            else self._amount_and_hour(merchant, weird)
        )
        timestamp = self._advance_clock(calibrated_hour)
        customer = self._choose_customer(timestamp.hour)
        if category is None:
            merchant = self._choose_merchant(customer)
        if self.behavior_profile or (category is None and self.calibration_pairs is None):
            amount_minor, _ = self._amount_and_hour(merchant, weird, customer)
        channel, rail = self._choose_channel()
        if force_channel:
            channel = force_channel
            rail = {"agentic_checkout": "card", "card_ecom": "card", "upi_p2p": "upi", "upi_collect": "upi", "wire": "wire"}[channel]
        device = self.devices[self.rng.choice(customer.device_ids)]
        beneficiary = self.beneficiaries[merchant.beneficiary_id]
        is_new_beneficiary = merchant.beneficiary_id not in customer.known_beneficiaries
        customer.known_beneficiaries.add(merchant.beneficiary_id)
        beneficiary.sender_ids.add(customer.customer_id)
        beneficiary.payment_count += 1
        device.last_seen = timestamp

        mandate_chain = None
        agent_trace = None
        if channel == "agentic_checkout":
            qty = 1
            item = {
                "sku": f"{merchant.sku_prefix}-{self.rng.randrange(10, 999):03d}",
                "name": merchant.item_name,
                "price_minor": amount_minor,
                "qty": qty,
            }
            budget_minor = max(amount_minor, round(amount_minor * self.rng.uniform(1.05, 1.35)))
            presence = "NOT_PRESENT" if self.rng.random() < 0.65 else "PRESENT"
            mandate_chain = build_chain(
                self.registry,
                user_id=customer.customer_id,
                merchant_id=merchant.merchant_id,
                agent_id="agent:shopper-v3",
                category=merchant.category,
                budget_minor=budget_minor,
                mcc=merchant.mcc,
                prompt_playback=f"Find {merchant.category.replace('_', ' ')} under INR {budget_minor / 100:.0f}",
                signed_at=timestamp - timedelta(seconds=self.rng.randint(30, 3_600)),
                items=[item],
                payee_did=merchant.payee_did,
                presence_flag=presence,
                delegation_depth=self.rng.choices((1, 2, 3), (0.25, 0.65, 0.10))[0],
            )
            agent_trace = {
                "tool_calls": [
                    {"t": 0, "tool": "web_search", "topic": merchant.category.replace("_", " ")},
                    {"t": 1, "tool": "fetch_page", "topic": merchant.category.replace("_", " ")},
                    {"t": 3, "tool": "create_payment_mandate", "topic": merchant.category.replace("_", " ")},
                ],
                "pages_read": [
                    {
                        "url": f"https://merchant-{merchant.merchant_id}.example/product",
                        "hidden_text_present": False,
                        "hidden_text_sample": "",
                        "zero_width_chars": 0,
                        "invisible_css": False,
                    }
                ],
                "negotiation_seconds": round(self.rng.uniform(2.5, 28.0), 3),
                "human_present": presence == "PRESENT",
            }

        self.event_counter += 1
        event = {
            "event_id": f"evt_{self.event_counter:08d}",
            "ts": _iso_z(timestamp),
            "channel": channel,
            "rail": rail,
            "mandate_chain": mandate_chain,
            "agent_trace": agent_trace,
            "txn": {
                "amount_minor": amount_minor,
                "currency": "INR",
                "mcc": merchant.mcc,
                "customer_id": customer.customer_id,
                "device_id": device.device_id,
                "geo": {
                    "lat": round(customer.home_geo[0] + self.rng.uniform(-0.05, 0.05), 6),
                    "lon": round(customer.home_geo[1] + self.rng.uniform(-0.05, 0.05), 6),
                },
                "hour_local": timestamp.hour,
                "beneficiary_id": merchant.beneficiary_id,
                "is_new_beneficiary": is_new_beneficiary,
                "beneficiary_in_degree": beneficiary.in_degree,
                "beneficiary_payment_count": beneficiary.payment_count,
                "device_age_days": (timestamp - device.first_seen).days,
                "customer_txn_count": len(customer.txn_history),
                "customer_avg_ticket_minor": round(customer.avg_ticket_minor),
                "merchant_risk": round(merchant.risk, 6),
                "merchant_id": merchant.merchant_id,
                "market_median_minor": merchant.market_median_minor,
            },
            "label": {
                "is_fraud": False,
                "atlas_id": None,
                "attack_primitives": [],
                "attack_family": None,
                "arena_round": None,
                "split": split,
            },
            "scores": _scores(),
            "decision": _decision(),
        }
        customer.txn_history.append(
            {"ts": event["ts"], "amount_minor": amount_minor, "merchant_id": merchant.merchant_id}
        )
        if len(customer.txn_history) > 200:
            del customer.txn_history[:-200]
        if not self.behavior_profile:
            customer.avg_ticket_minor = 0.97 * customer.avg_ticket_minor + 0.03 * amount_minor
        customer.merchant_affinity[merchant.merchant_id] = customer.merchant_affinity.get(merchant.merchant_id, 0) + 1
        return event

    def iter_legitimate(self, count: int, *, split: str = "A") -> Iterator[dict[str, Any]]:
        self._prepare_activity_quotas(count)
        for _ in range(count):
            yield self.generate_legitimate(split=split)

    def generate_event(self, *, fraud_rate: float = 0.0055) -> dict[str, Any]:
        if not 0 <= fraud_rate <= 1:
            raise ValueError("fraud_rate must be between zero and one")
        if self.rng.random() >= fraud_rate:
            return self.generate_legitimate()
        from sim.attacks import attacks_for_split, apply_attack

        attack = self.rng.choice(attacks_for_split(("A", "B")))
        return apply_attack(
            self.generate_legitimate(force_channel=attack.base_channel),
            attack,
            registry=self.registry,
        )

    def generate(self, count: int, *, fraud_rate: float = 0.0055) -> list[dict[str, Any]]:
        return [self.generate_event(fraud_rate=fraud_rate) for _ in range(count)]

    def state_summary(self) -> dict[str, int]:
        return {
            "events": self.event_counter,
            "customers_with_history": sum(bool(customer.txn_history) for customer in self.customers.values()),
            "beneficiary_edges": sum(beneficiary.in_degree for beneficiary in self.beneficiaries.values()),
            "beneficiary_payments": sum(beneficiary.payment_count for beneficiary in self.beneficiaries.values()),
        }


def event_rows(events: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for event in events:
        yield {
            "amount_minor": event["txn"]["amount_minor"],
            "hour_local": event["txn"]["hour_local"],
        }


def generate_population_dataset(
    output: Path,
    *,
    events: int = 50_000,
    seed: int = 20260827,
    fraud_rate: float = 0.0055,
    benign_adversarial_rate: float = 0.02,
    behavior_profile: BehaviorProfile | None = None,
    evaluation_preregistration_sha256: str | None = None,
) -> dict[str, Any]:
    from jsonschema import Draft202012Validator, FormatChecker

    from sim.attacks import all_attacks, apply_attack
    from sim.mandate_twin import DETERMINISTIC_SIGNING_AVAILABLE

    if events < 1 or not 0 <= fraud_rate <= 1 or not 0 <= benign_adversarial_rate <= 1:
        raise ValueError("invalid event count or rate")
    if evaluation_preregistration_sha256 is not None and not (
        isinstance(evaluation_preregistration_sha256, str)
        and len(evaluation_preregistration_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in evaluation_preregistration_sha256
        )
    ):
        raise ValueError("evaluation preregistration SHA-256 is invalid")
    if fraud_rate + benign_adversarial_rate > 1:
        raise ValueError("fraud and benign-adversarial rates exceed one")
    if not DETERMINISTIC_SIGNING_AVAILABLE:
        raise RuntimeError("deterministic population output requires cryptography >= 43")

    fraud_count = round(events * fraud_rate)
    benign_count = round(events * benign_adversarial_rate)
    attacks = all_attacks(include_heldout=True)
    assigned = [attacks[index % len(attacks)] for index in range(fraud_count)]
    early_fraud: list[Any] = []
    late_fraud: list[Any] = []
    frozen_fraud: list[Any] = []
    for attack in attacks:
        occurrences = [item for item in assigned if item is attack]
        if attack.split in {"C", "D"}:
            frozen_fraud.extend(occurrences)
            continue
        cut = max(1, round(len(occurrences) * 0.75))
        early_fraud.extend(occurrences[:cut])
        late_fraud.extend(occurrences[cut:])
    legitimate_count = events - fraud_count - benign_count
    early_legitimate = round(legitimate_count * 0.75)
    early: list[tuple[str, Any]] = [("fraud", attack) for attack in early_fraud]
    early.extend(("legitimate", None) for _ in range(early_legitimate))
    late: list[tuple[str, Any]] = [("fraud", attack) for attack in late_fraud]
    late.extend(("legitimate", None) for _ in range(legitimate_count - early_legitimate))
    frozen: list[tuple[str, Any]] = [("fraud", attack) for attack in frozen_fraud]
    simulator = PopulationSimulator(
        seed=seed,
        customers=(
            max(1, round(events / behavior_profile.activity_mean_transactions))
            if behavior_profile
            else 800
        ),
        behavior_profile=behavior_profile,
    )
    simulator._prepare_activity_quotas(events)
    for phase in (early, late, frozen):
        simulator.rng.shuffle(phase)
    schedule = early + late + frozen + [("E", None) for _ in range(benign_count)]
    schema = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for kind, attack in schedule:
            if kind == "fraud":
                base = simulator.generate_legitimate(force_channel=attack.base_channel)
                event = apply_attack(base, attack, registry=simulator.registry)
                simulator.synchronize_clock(event)
            elif kind == "E":
                event = simulator.generate_legitimate(weird=True)
            else:
                event = simulator.generate_legitimate()
            validator.validate(event)
            line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            split_counts[event["label"]["split"]] += 1
            if event["label"]["attack_family"]:
                family_counts[event["label"]["attack_family"]] += 1
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "events": events,
        "fraud_events": fraud_count,
        "fraud_rate": fraud_count / events,
        "benign_adversarial_events": benign_count,
        "split_counts": dict(sorted(split_counts.items())),
        "attack_family_counts": dict(sorted(family_counts.items())),
        "sha256": digest.hexdigest(),
        "synthetic_pii": True,
        "api_calls": 0,
        "behavior_profile_sha256": behavior_profile.sha256 if behavior_profile else None,
        "evaluation_preregistration_sha256": evaluation_preregistration_sha256,
        **simulator.registry.manifest_fields(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--events", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--fraud-rate", type=float, default=0.0055)
    parser.add_argument("--benign-adversarial-rate", type=float, default=0.02)
    parser.add_argument("--behavior-profile", type=Path)
    parser.add_argument("--evaluation-preregistration", type=Path)
    args = parser.parse_args()
    behavior_profile = None
    if args.behavior_profile:
        profile_payload = json.loads(args.behavior_profile.read_text(encoding="utf-8"))
        behavior_profile = BehaviorProfile.from_dict(
            profile_payload.get("behavior_profile", profile_payload)
        )
    evaluation_preregistration_sha256 = None
    if args.evaluation_preregistration:
        from scripts.release import (
            ReleaseInvalid,
            _require_evaluation_preregistration,
        )

        path = args.evaluation_preregistration
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError("evaluation preregistration must be an immutable regular file")
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("evaluation preregistration does not bind this generator")
        artifact_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            _require_evaluation_preregistration(
                payload,
                artifact_sha256=artifact_sha256,
                project_root=ROOT,
            )
        except ReleaseInvalid as error:
            raise RuntimeError(
                "evaluation preregistration does not bind this generator"
            ) from error
        lineage = payload.get("lineage") if isinstance(payload, dict) else None
        remediation = payload.get("remediation") if isinstance(payload, dict) else None
        generator = payload.get("generator") if isinstance(payload, dict) else None
        arena = payload.get("arena") if isinstance(payload, dict) else None
        evaluation = payload.get("evaluation") if isinstance(payload, dict) else None
        custody = payload.get("custody") if isinstance(payload, dict) else None
        if (
            set(payload)
            != {
                "schema_version",
                "artifact",
                "status",
                "claim_scope",
                "pristine_heldout_family",
                "independent_holdout",
                "fresh_rows_only",
                "lineage",
                "remediation",
                "generator",
                "arena",
                "evaluation",
                "custody",
                "bindings",
            }
            or path.name != EVALUATION_PREREGISTRATION.name
            or payload.get("schema_version") != 4
            or payload.get("artifact") != "evaluation_v4_preregistration"
            or payload.get("status") != "predeclared_before_post_remediation_regression"
            or payload.get("claim_scope") != "post_remediation_regression"
            or payload.get("pristine_heldout_family") is not False
            or payload.get("independent_holdout") is not False
            or payload.get("fresh_rows_only") is not True
            or lineage != V4_LINEAGE
            or remediation != V4_REMEDIATION
            or generator != V4_GENERATOR
            or arena != V4_ARENA
            or evaluation != V4_EVALUATION
            or custody != V4_CUSTODY
            or args.seed != V4_GENERATOR["seed"]
            or args.events != V4_GENERATOR["events"]
            or args.fraud_rate != V4_GENERATOR["fraud_rate"]
            or args.benign_adversarial_rate
            != V4_GENERATOR["benign_adversarial_rate"]
        ):
            raise RuntimeError("evaluation preregistration does not bind this generator")
        evaluation_preregistration_sha256 = artifact_sha256
    manifest = generate_population_dataset(
        args.output,
        events=args.events,
        seed=args.seed,
        fraud_rate=args.fraud_rate,
        benign_adversarial_rate=args.benign_adversarial_rate,
        behavior_profile=behavior_profile,
        evaluation_preregistration_sha256=evaluation_preregistration_sha256,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
