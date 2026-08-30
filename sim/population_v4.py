"""Schema-4 checkerboard population profile and canonical generator."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sim.population import (
    COPULA_QUANTILE_PROBABILITIES,
    ROOT,
    PopulationSimulator,
)


CHECKERBOARD_BINS = 3
QUANTILE_PROBABILITIES = COPULA_QUANTILE_PROBABILITIES
STRATUM_ORDER = ("singleton", "same_day_repeat", "multiday_repeat")
STRATUM_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "singleton": (
        "active_start_fraction",
        "relative_log_amount_center",
        "circadian_phase_latent",
    ),
    "same_day_repeat": (
        "invoice_count",
        "active_start_fraction",
        "relative_log_amount_center",
        "eb_log_amount_scale",
        "eb_short_session_probability",
        "circadian_phase_latent",
        "eb_circadian_concentration",
    ),
    "multiday_repeat": (
        "invoice_count",
        "active_start_fraction",
        "active_duration_fraction",
        "relative_log_amount_center",
        "eb_log_amount_scale",
        "eb_short_session_probability",
        "circadian_phase_latent",
        "eb_circadian_concentration",
    ),
}
STRATUM_CELL_COUNTS = {
    name: CHECKERBOARD_BINS ** len(primitives)
    for name, primitives in STRATUM_PRIMITIVES.items()
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} fields differ from the frozen schema")


def checkerboard_marginals(
    masses: Iterable[float], dimension: int, *, bins: int = CHECKERBOARD_BINS
) -> tuple[tuple[float, ...], ...]:
    """Return each coordinate's discrete checkerboard marginal."""
    import numpy as np

    tensor = np.asarray(tuple(masses), dtype=float).reshape((bins,) * dimension)
    output = []
    for coordinate in range(dimension):
        axes = tuple(index for index in range(dimension) if index != coordinate)
        output.append(tuple(map(float, tensor.sum(axis=axes))))
    return tuple(output)


@dataclass(frozen=True)
class CheckerboardStratum:
    """Fixed-size fractional-tie checkerboard aggregate for one support stratum."""

    name: str
    customer_count: int
    primitive_quantiles: tuple[tuple[float, ...], ...]
    cell_masses: tuple[float, ...]

    def __post_init__(self) -> None:
        import numpy as np

        if self.name not in STRATUM_PRIMITIVES:
            raise ValueError("unknown checkerboard stratum")
        primitives = STRATUM_PRIMITIVES[self.name]
        if self.customer_count < 32:
            raise ValueError("every checkerboard stratum requires at least 32 customers")
        if (
            len(self.primitive_quantiles) != len(primitives)
            or any(len(values) != 32 for values in self.primitive_quantiles)
            or any(
                not math.isfinite(value)
                for values in self.primitive_quantiles
                for value in values
            )
            or any(
                left > right
                for values in self.primitive_quantiles
                for left, right in zip(values, values[1:])
            )
        ):
            raise ValueError("checkerboard marginal quantiles are invalid")
        masses = np.asarray(self.cell_masses, dtype=float)
        if (
            len(masses) != STRATUM_CELL_COUNTS[self.name]
            or not np.isfinite(masses).all()
            or (masses < 0).any()
            or not math.isclose(float(masses.sum()), 1.0, abs_tol=2e-10)
        ):
            raise ValueError("checkerboard masses are invalid")
        for marginal in checkerboard_marginals(masses, len(primitives)):
            if any(
                not math.isclose(value, 1 / CHECKERBOARD_BINS, abs_tol=2e-10)
                for value in marginal
            ):
                raise ValueError("checkerboard marginal thirds must be uniform")

    @property
    def primitives(self) -> tuple[str, ...]:
        return STRATUM_PRIMITIVES[self.name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "customers": self.customer_count,
            "primitives": list(self.primitives),
            "primitive_quantiles": {
                name: list(values)
                for name, values in zip(
                    self.primitives, self.primitive_quantiles, strict=True
                )
            },
            "cell_masses": list(self.cell_masses),
        }

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "CheckerboardStratum":
        _require_keys(
            value,
            {"customers", "primitives", "primitive_quantiles", "cell_masses"},
            f"{name} stratum",
        )
        primitives = STRATUM_PRIMITIVES.get(name)
        if primitives is None or tuple(value["primitives"]) != primitives:
            raise ValueError("checkerboard primitive order changed")
        quantiles = value["primitive_quantiles"]
        if set(quantiles) != set(primitives):
            raise ValueError("checkerboard primitive quantiles changed")
        return cls(
            name=name,
            customer_count=int(value["customers"]),
            primitive_quantiles=tuple(
                tuple(map(float, quantiles[primitive])) for primitive in primitives
            ),
            cell_masses=tuple(map(float, value["cell_masses"])),
        )


def _largest_remainder_counts(total: int, weights: Iterable[float]) -> tuple[int, ...]:
    """Allocate an integer total with stable, declared-order tie breaking."""
    weights_tuple = tuple(map(float, weights))
    if total < 0 or not weights_tuple or any(weight < 0 for weight in weights_tuple):
        raise ValueError("largest-remainder inputs are invalid")
    weight_sum = sum(weights_tuple)
    if weight_sum <= 0:
        raise ValueError("largest-remainder weights must have positive mass")
    targets = [total * weight / weight_sum for weight in weights_tuple]
    counts = [math.floor(target) for target in targets]
    remaining = total - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(targets[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def _residual_cell_counts(masses: Any, count: int, rng: Any) -> Any:
    """Residual-resample exact cell counts with no smoothing or pseudocounts."""
    import numpy as np

    probabilities = np.asarray(masses, dtype=float)
    expected = count * probabilities
    allocated = np.floor(expected).astype(int)
    remaining = count - int(allocated.sum())
    if remaining < 0:
        raise RuntimeError("checkerboard allocation exceeded its exact total")
    if remaining:
        residual = np.clip(expected - allocated, 0.0, None)
        residual_sum = float(residual.sum())
        if residual_sum <= 0:
            raise RuntimeError("checkerboard residual allocation has no mass")
        selected = rng.choice(
            len(probabilities), size=remaining, replace=True, p=residual / residual_sum
        )
        allocated += np.bincount(selected, minlength=len(probabilities))
    if int(allocated.sum()) != count:
        raise RuntimeError("checkerboard residual allocation changed its total")
    return allocated


@dataclass(frozen=True)
class BehaviorProfileV4:
    """Fixed-size schema-4 aggregate; contains no linked customer vectors or rows."""

    source_dataset: str
    source_partition: str
    source_customer_ids_sha256: str
    source_transactions_sha256: str
    source_customer_count: int
    source_transaction_count: int
    quantile_probabilities: tuple[float, ...]
    strata: tuple[CheckerboardStratum, ...]
    hour_weights: tuple[float, ...]
    calendar_weights: tuple[float, ...]
    observation_days: float
    session_gap_log_mean: float
    session_gap_log_std: float
    circadian_phase_cut: float
    amount_variance_prior_alpha: float
    amount_variance_prior_beta: float
    short_session_prior_alpha: float
    short_session_prior_beta: float

    def __post_init__(self) -> None:
        if self.source_partition != "calibration":
            raise ValueError("v4 profiles may only be fitted on calibration customers")
        if self.quantile_probabilities != QUANTILE_PROBABILITIES:
            raise ValueError("v4 profile must use the frozen 32 tail-dense knots")
        if tuple(stratum.name for stratum in self.strata) != STRATUM_ORDER:
            raise ValueError("checkerboard stratum order changed")
        if sum(stratum.customer_count for stratum in self.strata) != self.source_customer_count:
            raise ValueError("checkerboard stratum counts do not cover the source customers")
        if (
            self.source_customer_count < 96
            or self.source_transaction_count < self.source_customer_count
            or len(self.source_customer_ids_sha256) != 64
            or len(self.source_transactions_sha256) != 64
        ):
            raise ValueError("v4 source aggregate is invalid")
        if (
            len(self.hour_weights) != 24
            or any(weight < 0 or not math.isfinite(weight) for weight in self.hour_weights)
            or not math.isclose(sum(self.hour_weights), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("v4 profile needs 24 normalized hour weights")
        if (
            len(self.calendar_weights) != 12
            or any(weight < 0 or not math.isfinite(weight) for weight in self.calendar_weights)
            or not math.isclose(sum(self.calendar_weights), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("v4 profile needs 12 normalized calendar weights")
        renderer_scalars = (
            self.observation_days,
            self.session_gap_log_mean,
            self.session_gap_log_std,
            self.circadian_phase_cut,
            self.amount_variance_prior_alpha,
            self.amount_variance_prior_beta,
            self.short_session_prior_alpha,
            self.short_session_prior_beta,
        )
        if (
            not all(map(math.isfinite, renderer_scalars))
            or self.observation_days <= 1
            or self.session_gap_log_std < 0
            or not 0 <= self.circadian_phase_cut < 2 * math.pi
            or min(
                self.amount_variance_prior_alpha,
                self.amount_variance_prior_beta,
                self.short_session_prior_alpha,
                self.short_session_prior_beta,
            )
            <= 0
        ):
            raise ValueError("v4 renderer or empirical-Bayes aggregates are invalid")

    @property
    def activity_mean_transactions(self) -> float:
        return self.source_transaction_count / self.source_customer_count

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "source": {
                "dataset": self.source_dataset,
                "partition": self.source_partition,
                "customer_ids_sha256": self.source_customer_ids_sha256,
                "transactions_sha256": self.source_transactions_sha256,
                "customers": self.source_customer_count,
                "transactions": self.source_transaction_count,
            },
            "checkerboard": {
                "bins": CHECKERBOARD_BINS,
                "quantile_probabilities": list(self.quantile_probabilities),
                "strata": {stratum.name: stratum.as_dict() for stratum in self.strata},
                "smoothing": False,
                "pseudocounts": False,
                "stored_rank_rows": False,
            },
            "renderer": {
                "hour_weights": list(self.hour_weights),
                "calendar_weights": list(self.calendar_weights),
                "observation_days": self.observation_days,
                "session_gap_log_mean": self.session_gap_log_mean,
                "session_gap_log_std": self.session_gap_log_std,
                "circadian_phase_cut": self.circadian_phase_cut,
            },
            "empirical_bayes": {
                "amount_variance_prior_alpha": self.amount_variance_prior_alpha,
                "amount_variance_prior_beta": self.amount_variance_prior_beta,
                "short_session_prior_alpha": self.short_session_prior_alpha,
                "short_session_prior_beta": self.short_session_prior_beta,
            },
            "privacy": {
                "fixed_size_aggregate": True,
                "contains_customer_ids": False,
                "contains_raw_rows": False,
                "contains_rank_rows": False,
                "contains_evaluator_feature_rows": False,
                "differential_privacy_claimed": False,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorProfileV4":
        _require_keys(
            value,
            {
                "schema_version",
                "source",
                "checkerboard",
                "renderer",
                "empirical_bayes",
                "privacy",
            },
            "v4 profile",
        )
        if value["schema_version"] != 4:
            raise ValueError("only frozen schema-4 behavior profiles are accepted")
        source = value["source"]
        checkerboard = value["checkerboard"]
        renderer = value["renderer"]
        empirical_bayes = value["empirical_bayes"]
        privacy = value["privacy"]
        _require_keys(
            source,
            {
                "dataset",
                "partition",
                "customer_ids_sha256",
                "transactions_sha256",
                "customers",
                "transactions",
            },
            "v4 source",
        )
        _require_keys(
            checkerboard,
            {
                "bins",
                "quantile_probabilities",
                "strata",
                "smoothing",
                "pseudocounts",
                "stored_rank_rows",
            },
            "v4 checkerboard",
        )
        _require_keys(
            renderer,
            {
                "hour_weights",
                "calendar_weights",
                "observation_days",
                "session_gap_log_mean",
                "session_gap_log_std",
                "circadian_phase_cut",
            },
            "v4 renderer",
        )
        _require_keys(
            empirical_bayes,
            {
                "amount_variance_prior_alpha",
                "amount_variance_prior_beta",
                "short_session_prior_alpha",
                "short_session_prior_beta",
            },
            "v4 empirical-Bayes",
        )
        expected_privacy = {
            "fixed_size_aggregate": True,
            "contains_customer_ids": False,
            "contains_raw_rows": False,
            "contains_rank_rows": False,
            "contains_evaluator_feature_rows": False,
            "differential_privacy_claimed": False,
        }
        if privacy != expected_privacy:
            raise ValueError("v4 privacy declaration changed")
        if (
            checkerboard["bins"] != CHECKERBOARD_BINS
            or checkerboard["smoothing"] is not False
            or checkerboard["pseudocounts"] is not False
            or checkerboard["stored_rank_rows"] is not False
            or set(checkerboard["strata"]) != set(STRATUM_ORDER)
        ):
            raise ValueError("v4 checkerboard contract changed")
        return cls(
            source_dataset=str(source["dataset"]),
            source_partition=str(source["partition"]),
            source_customer_ids_sha256=str(source["customer_ids_sha256"]),
            source_transactions_sha256=str(source["transactions_sha256"]),
            source_customer_count=int(source["customers"]),
            source_transaction_count=int(source["transactions"]),
            quantile_probabilities=tuple(map(float, checkerboard["quantile_probabilities"])),
            strata=tuple(
                CheckerboardStratum.from_dict(name, checkerboard["strata"][name])
                for name in STRATUM_ORDER
            ),
            hour_weights=tuple(map(float, renderer["hour_weights"])),
            calendar_weights=tuple(map(float, renderer["calendar_weights"])),
            observation_days=float(renderer["observation_days"]),
            session_gap_log_mean=float(renderer["session_gap_log_mean"]),
            session_gap_log_std=float(renderer["session_gap_log_std"]),
            circadian_phase_cut=float(renderer["circadian_phase_cut"]),
            amount_variance_prior_alpha=float(
                empirical_bayes["amount_variance_prior_alpha"]
            ),
            amount_variance_prior_beta=float(
                empirical_bayes["amount_variance_prior_beta"]
            ),
            short_session_prior_alpha=float(
                empirical_bayes["short_session_prior_alpha"]
            ),
            short_session_prior_beta=float(
                empirical_bayes["short_session_prior_beta"]
            ),
        )

    def _sample_stratum(self, stratum: CheckerboardStratum, count: int, rng: Any) -> list[dict[str, float]]:
        import numpy as np

        if count == 0:
            return []
        allocated = _residual_cell_counts(stratum.cell_masses, count, rng)
        cell_indices = np.repeat(np.arange(len(allocated)), allocated)
        rng.shuffle(cell_indices)
        cells = np.column_stack(
            np.unravel_index(cell_indices, (CHECKERBOARD_BINS,) * len(stratum.primitives))
        )
        uniforms = (cells + rng.random(cells.shape)) / CHECKERBOARD_BINS
        probabilities = np.asarray(self.quantile_probabilities, dtype=float)
        sampled = np.column_stack(
            [
                np.interp(uniforms[:, index], probabilities, values)
                for index, values in enumerate(stratum.primitive_quantiles)
            ]
        )
        output: list[dict[str, float]] = []
        supported_hours = [hour for hour, weight in enumerate(self.hour_weights) if weight > 0]
        for values in sampled:
            row = dict(zip(stratum.primitives, map(float, values), strict=True))
            if stratum.name == "singleton":
                row.update(
                    {
                        "invoice_count": 1.0,
                        "active_duration_fraction": 0.0,
                        "eb_log_amount_scale": 0.0,
                        "eb_short_session_probability": 0.0,
                        "eb_circadian_concentration": 0.0,
                    }
                )
            else:
                row["invoice_count"] = float(max(2, round(row["invoice_count"])))
                row["eb_log_amount_scale"] = max(1e-6, row["eb_log_amount_scale"])
                row["eb_short_session_probability"] = min(
                    1.0, max(0.0, row["eb_short_session_probability"])
                )
                row["eb_circadian_concentration"] = max(
                    0.0, row["eb_circadian_concentration"]
                )
                if stratum.name == "same_day_repeat":
                    row["active_duration_fraction"] = 0.0
            row["active_start_fraction"] = min(
                1.0, max(0.0, row["active_start_fraction"])
            )
            if stratum.name == "multiday_repeat":
                minimum = math.nextafter(1.0 / self.observation_days, math.inf)
                row["active_start_fraction"] = min(
                    max(0.0, 1.0 - minimum), row["active_start_fraction"]
                )
                capacity = 1.0 - row["active_start_fraction"]
                row["active_duration_fraction"] = min(
                    capacity, max(minimum, row["active_duration_fraction"])
                )
                if (
                    row["active_duration_fraction"] <= 0
                    or row["active_start_fraction"]
                    + row["active_duration_fraction"]
                    > 1
                ):
                    raise RuntimeError("multiday support projection is invalid")
            phase = (
                row["circadian_phase_latent"] + self.circadian_phase_cut
            ) % (2 * math.pi)
            if stratum.name == "singleton":
                hour = min(
                    supported_hours,
                    key=lambda candidate: abs(
                        math.atan2(
                            math.sin(phase - 2 * math.pi * candidate / 24),
                            math.cos(phase - 2 * math.pi * candidate / 24),
                        )
                    ),
                )
                phase = 2 * math.pi * hour / 24
                row["fixed_hour"] = float(hour)
            row["circadian_phase_sin"] = math.sin(phase)
            row["circadian_phase_cos"] = math.cos(phase)
            output.append(row)
        return output

    def sample_primitives(self, count: int, *, seed: int) -> list[dict[str, float]]:
        """Sample exact support strata and checkerboard cells deterministically."""
        import numpy as np

        if count < 0:
            raise ValueError("sample count must be nonnegative")
        if count == 0:
            return []
        rng = np.random.default_rng(seed)
        stratum_counts = _largest_remainder_counts(
            count, (stratum.customer_count for stratum in self.strata)
        )
        output = [
            row
            for stratum, stratum_count in zip(self.strata, stratum_counts, strict=True)
            for row in self._sample_stratum(stratum, stratum_count, rng)
        ]
        permutation = rng.permutation(len(output))
        return [output[int(index)] for index in permutation]


class PopulationSimulatorV4(PopulationSimulator):
    """V3 event builder with schema-4 structural quota preservation."""

    def _prepare_activity_quotas(self, events: int) -> None:
        if not self.behavior_profile or self._quota_prepared_for == events:
            return
        population = list(self.customers.values())
        minima = [1 if round(customer.activity_weight) == 1 else 2 for customer in population]
        minimum_total = sum(minima)
        if events < minimum_total:
            raise ValueError("event quota cannot preserve v4 structural strata")
        sampled_quotas = [round(customer.activity_weight) for customer in population]
        if events == sum(sampled_quotas):
            quotas = sampled_quotas
        else:
            quotas = []
        extras = events - minimum_total
        repeat_indices = [index for index, minimum in enumerate(minima) if minimum == 2]
        raw = [max(0.0, customer.activity_weight - minimum) for customer, minimum in zip(population, minima, strict=True)]
        total = sum(raw)
        if extras and not repeat_indices:
            raise ValueError("extra events cannot be assigned without changing singleton support")
        if not quotas:
            if total:
                scaled = [extras * value / total for value in raw]
            else:
                scaled = [0.0] * len(population)
                if repeat_indices:
                    for index in repeat_indices:
                        scaled[index] = extras / len(repeat_indices)
            quotas = [minimum + math.floor(value) for minimum, value in zip(minima, scaled, strict=True)]
            remaining = events - sum(quotas)
            order = sorted(
                repeat_indices,
                key=lambda index: (
                    -(scaled[index] - math.floor(scaled[index])),
                    population[index].customer_id,
                ),
            )
            for index in order[:remaining]:
                quotas[index] += 1
        if sum(quotas) != events:
            raise RuntimeError("v4 structural quota allocation changed the event total")
        for customer, quota in zip(population, quotas, strict=True):
            customer.remaining_quota = quota
        schedules = [
            self._customer_schedule(customer, quota)
            for customer, quota in zip(population, quotas, strict=True)
        ]
        self._schedule = list(heapq.merge(*schedules))
        if len(self._schedule) != events:
            raise RuntimeError("v4 schedules did not preserve the exact event quota")
        self._schedule_index = 0
        self._quota_prepared_for = events


def generate_population_dataset_v4(
    output: Path,
    *,
    events: int = 50_000,
    seed: int = 20260827,
    fraud_rate: float = 0.0055,
    benign_adversarial_rate: float = 0.02,
    behavior_profile: BehaviorProfileV4 | None = None,
) -> dict[str, Any]:
    """Canonical dataset generation using the existing event builder and v4 profile."""
    from jsonschema import Draft202012Validator, FormatChecker

    from sim.attacks import all_attacks, apply_attack
    from sim.mandate_twin import DETERMINISTIC_SIGNING_AVAILABLE

    if events < 1 or not 0 <= fraud_rate <= 1 or not 0 <= benign_adversarial_rate <= 1:
        raise ValueError("invalid event count or rate")
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
    simulator = PopulationSimulatorV4(
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
            digest.update(line.encode())
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
        "behavior_profile_schema_version": 4 if behavior_profile else None,
        "behavior_profile_sha256": behavior_profile.sha256 if behavior_profile else None,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--events", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fraud-rate", type=float, default=0.0055)
    parser.add_argument("--benign-adversarial-rate", type=float, default=0.02)
    parser.add_argument("--behavior-profile", type=Path)
    args = parser.parse_args()
    behavior_profile = None
    if args.behavior_profile:
        payload = json.loads(args.behavior_profile.read_text(encoding="utf-8"))
        behavior_profile = BehaviorProfileV4.from_dict(
            payload.get("behavior_profile", payload)
        )
    result = generate_population_dataset_v4(
        args.output,
        events=args.events,
        seed=args.seed,
        fraud_rate=args.fraud_rate,
        benign_adversarial_rate=args.benign_adversarial_rate,
        behavior_profile=behavior_profile,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
