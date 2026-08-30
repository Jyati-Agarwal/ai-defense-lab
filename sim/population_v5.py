"""Schema-5 checkerboard population with a conditional same-day atom."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from sim.population import ROOT
from sim.population_v4 import (
    CHECKERBOARD_BINS,
    QUANTILE_PROBABILITIES,
    STRATUM_PRIMITIVES as V4_STRATUM_PRIMITIVES,
    CheckerboardStratum,
    PopulationSimulatorV4,
    _canonical_hash,
    _largest_remainder_counts,
    _residual_cell_counts,
)


CHECKERBOARD_STRATA = ("singleton", "multiday_repeat")
STRATUM_PRIMITIVES = {
    name: V4_STRATUM_PRIMITIVES[name] for name in CHECKERBOARD_STRATA
}
STRATUM_CELL_COUNTS = {
    name: CHECKERBOARD_BINS ** len(primitives)
    for name, primitives in STRATUM_PRIMITIVES.items()
}
MIN_CHECKERBOARD_CUSTOMERS = 32
ATOM_SELECTION_NAMESPACE = "entity-fidelity-v5-same-day-atom"
ATOM_SELECTION_RULE = (
    "select exactly round(q*repeat_count) repeat positions by ascending "
    "SHA256(seed:entity-fidelity-v5-same-day-atom:position), ties by position"
)


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} fields differ from the frozen schema")


@dataclass(frozen=True)
class BehaviorProfileV5:
    """Two fixed checkerboards plus a calibration-fitted repeat same-day atom."""

    source_dataset: str
    source_partition: str
    source_customer_ids_sha256: str
    source_transactions_sha256: str
    source_customer_count: int
    source_transaction_count: int
    repeat_customer_count: int
    same_day_repeat_probability: float
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
            raise ValueError("v5 profiles may only be fitted on calibration customers")
        if self.quantile_probabilities != QUANTILE_PROBABILITIES:
            raise ValueError("v5 profile must use the frozen 32 tail-dense knots")
        if tuple(stratum.name for stratum in self.strata) != CHECKERBOARD_STRATA:
            raise ValueError("v5 checkerboard stratum order changed")
        if any(
            stratum.customer_count < MIN_CHECKERBOARD_CUSTOMERS
            for stratum in self.strata
        ):
            raise ValueError("v5 checkerboards require at least 32 customers each")
        singleton_count = self.strata[0].customer_count
        multiday_count = self.strata[1].customer_count
        if (
            self.repeat_customer_count < multiday_count
            or singleton_count + self.repeat_customer_count
            != self.source_customer_count
            or self.source_customer_count < 2 * MIN_CHECKERBOARD_CUSTOMERS
            or self.source_transaction_count < self.source_customer_count
            or len(self.source_customer_ids_sha256) != 64
            or len(self.source_transactions_sha256) != 64
        ):
            raise ValueError("v5 source or structural counts are invalid")
        expected_q = (
            self.repeat_customer_count - multiday_count
        ) / self.repeat_customer_count
        if (
            not math.isfinite(self.same_day_repeat_probability)
            or not 0 <= self.same_day_repeat_probability <= 1
            or not math.isclose(
                self.same_day_repeat_probability, expected_q, abs_tol=1e-15
            )
        ):
            raise ValueError("v5 same-day repeat atom is inconsistent with source counts")
        if (
            len(self.hour_weights) != 24
            or any(weight < 0 or not math.isfinite(weight) for weight in self.hour_weights)
            or not math.isclose(sum(self.hour_weights), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("v5 profile needs 24 normalized hour weights")
        if (
            len(self.calendar_weights) != 12
            or any(
                weight < 0 or not math.isfinite(weight)
                for weight in self.calendar_weights
            )
            or not math.isclose(sum(self.calendar_weights), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("v5 profile needs 12 normalized calendar weights")
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
            raise ValueError("v5 renderer or empirical-Bayes aggregates are invalid")

    @property
    def activity_mean_transactions(self) -> float:
        return self.source_transaction_count / self.source_customer_count

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 5,
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
                "strata": {
                    stratum.name: stratum.as_dict() for stratum in self.strata
                },
                "smoothing": False,
                "pseudocounts": False,
                "stored_rank_rows": False,
            },
            "same_day_repeat_atom": {
                "condition": "invoice_count>=2",
                "repeat_customers": self.repeat_customer_count,
                "probability": self.same_day_repeat_probability,
                "allocation": "exact round(probability*sampled_repeat_count)",
                "selection": ATOM_SELECTION_RULE,
                "projection": "set active_duration_fraction=0 only on selected repeats",
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
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorProfileV5":
        _require_keys(
            value,
            {
                "schema_version",
                "source",
                "checkerboard",
                "same_day_repeat_atom",
                "renderer",
                "empirical_bayes",
                "privacy",
            },
            "v5 profile",
        )
        if value["schema_version"] != 5:
            raise ValueError("only frozen schema-5 behavior profiles are accepted")
        source = value["source"]
        checkerboard = value["checkerboard"]
        atom = value["same_day_repeat_atom"]
        renderer = value["renderer"]
        empirical_bayes = value["empirical_bayes"]
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
            "v5 source",
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
            "v5 checkerboard",
        )
        _require_keys(
            atom,
            {
                "condition",
                "repeat_customers",
                "probability",
                "allocation",
                "selection",
                "projection",
            },
            "v5 same-day atom",
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
            "v5 renderer",
        )
        _require_keys(
            empirical_bayes,
            {
                "amount_variance_prior_alpha",
                "amount_variance_prior_beta",
                "short_session_prior_alpha",
                "short_session_prior_beta",
            },
            "v5 empirical-Bayes",
        )
        expected_privacy = {
            "fixed_size_aggregate": True,
            "contains_customer_ids": False,
            "contains_raw_rows": False,
            "contains_rank_rows": False,
            "contains_evaluator_feature_rows": False,
            "differential_privacy_claimed": False,
        }
        if value["privacy"] != expected_privacy:
            raise ValueError("v5 privacy declaration changed")
        if (
            checkerboard["bins"] != CHECKERBOARD_BINS
            or checkerboard["smoothing"] is not False
            or checkerboard["pseudocounts"] is not False
            or checkerboard["stored_rank_rows"] is not False
            or set(checkerboard["strata"]) != set(CHECKERBOARD_STRATA)
            or atom["condition"] != "invoice_count>=2"
            or atom["allocation"]
            != "exact round(probability*sampled_repeat_count)"
            or atom["selection"] != ATOM_SELECTION_RULE
            or atom["projection"]
            != "set active_duration_fraction=0 only on selected repeats"
        ):
            raise ValueError("v5 structural contract changed")
        return cls(
            source_dataset=str(source["dataset"]),
            source_partition=str(source["partition"]),
            source_customer_ids_sha256=str(source["customer_ids_sha256"]),
            source_transactions_sha256=str(source["transactions_sha256"]),
            source_customer_count=int(source["customers"]),
            source_transaction_count=int(source["transactions"]),
            repeat_customer_count=int(atom["repeat_customers"]),
            same_day_repeat_probability=float(atom["probability"]),
            quantile_probabilities=tuple(
                map(float, checkerboard["quantile_probabilities"])
            ),
            strata=tuple(
                CheckerboardStratum.from_dict(
                    name, checkerboard["strata"][name]
                )
                for name in CHECKERBOARD_STRATA
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

    def _sample_checkerboard(
        self, stratum: CheckerboardStratum, count: int, rng: Any
    ) -> list[dict[str, float]]:
        import numpy as np

        if count == 0:
            return []
        allocated = _residual_cell_counts(stratum.cell_masses, count, rng)
        cell_indices = np.repeat(np.arange(len(allocated)), allocated)
        rng.shuffle(cell_indices)
        cells = np.column_stack(
            np.unravel_index(
                cell_indices,
                (CHECKERBOARD_BINS,) * len(stratum.primitives),
            )
        )
        uniforms = (cells + rng.random(cells.shape)) / CHECKERBOARD_BINS
        probabilities = np.asarray(self.quantile_probabilities, dtype=float)
        sampled = np.column_stack(
            [
                np.interp(uniforms[:, index], probabilities, values)
                for index, values in enumerate(stratum.primitive_quantiles)
            ]
        )
        output = []
        supported_hours = [
            hour for hour, weight in enumerate(self.hour_weights) if weight > 0
        ]
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
                row["eb_log_amount_scale"] = max(
                    1e-6, row["eb_log_amount_scale"]
                )
                row["eb_short_session_probability"] = min(
                    1.0, max(0.0, row["eb_short_session_probability"])
                )
                row["eb_circadian_concentration"] = max(
                    0.0, row["eb_circadian_concentration"]
                )
            row["active_start_fraction"] = min(
                1.0, max(0.0, row["active_start_fraction"])
            )
            if stratum.name == "multiday_repeat":
                minimum = math.nextafter(
                    1.0 / self.observation_days, math.inf
                )
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
                    raise RuntimeError("v5 multiday support projection is invalid")
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
        """Draw two checkerboards and project an exact conditional repeat atom."""
        import numpy as np

        if count < 0:
            raise ValueError("sample count must be nonnegative")
        if count == 0:
            return []
        rng = np.random.default_rng(seed)
        singleton_count, repeat_count = _largest_remainder_counts(
            count,
            (self.strata[0].customer_count, self.repeat_customer_count),
        )
        singleton_rows = self._sample_checkerboard(
            self.strata[0], singleton_count, rng
        )
        repeat_rows = self._sample_checkerboard(
            self.strata[1], repeat_count, rng
        )
        atom_count = round(self.same_day_repeat_probability * repeat_count)
        atom_order = sorted(
            range(repeat_count),
            key=lambda index: (
                hashlib.sha256(
                    f"{seed}:{ATOM_SELECTION_NAMESPACE}:{index}".encode()
                ).digest(),
                index,
            ),
        )
        for index in atom_order[:atom_count]:
            repeat_rows[index]["active_duration_fraction"] = 0.0
        output = singleton_rows + repeat_rows
        permutation = rng.permutation(len(output))
        return [output[int(index)] for index in permutation]


class PopulationSimulatorV5(PopulationSimulatorV4):
    """Existing stateful event builder with V5 structural quota preservation."""

    def _customer_schedule(
        self, customer: Any, quota: int
    ) -> list[tuple[Any, str]]:
        schedule = super()._customer_schedule(customer, quota)
        if quota <= 1 or customer.active_duration_fraction != 0.0:
            return schedule
        seconds_per_day = 86_400
        if quota > seconds_per_day:
            raise ValueError(
                "same-day repeat quota exceeds strict one-second UTC-day capacity"
            )
        timestamps = [timestamp for timestamp, _ in schedule]
        midnight = timestamps[0].replace(hour=0, minute=0, second=0, microsecond=0)
        first_second = int((timestamps[0] - midnight).total_seconds())
        start_second = min(first_second, seconds_per_day - quota)
        available = seconds_per_day - 1 - start_second
        span = max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())
        scale = min(1.0, available / span) if span else 0.0
        offsets: list[int] = []
        for index, timestamp in enumerate(timestamps):
            preferred = round(
                (timestamp - timestamps[0]).total_seconds() * scale
            )
            lower = offsets[-1] + 1 if offsets else 0
            upper = available - (quota - 1 - index)
            offsets.append(min(upper, max(lower, preferred)))
        projected = [
            midnight + timedelta(seconds=start_second + offset)
            for offset in offsets
        ]
        if (
            len({timestamp.date() for timestamp in projected}) != 1
            or any(
                right <= left
                for left, right in zip(projected, projected[1:])
            )
        ):
            raise RuntimeError("v5 same-day schedule projection is invalid")
        return [(timestamp, customer.customer_id) for timestamp in projected]


def generate_population_dataset_v5(
    output: Path,
    *,
    events: int = 50_000,
    seed: int = 20260827,
    fraud_rate: float = 0.0055,
    benign_adversarial_rate: float = 0.02,
    behavior_profile: BehaviorProfileV5 | None = None,
) -> dict[str, Any]:
    """Run the canonical exact-event generator through the V5 scheduler."""
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
    early: list[tuple[str, Any]] = [
        ("fraud", attack) for attack in early_fraud
    ]
    early.extend(("legitimate", None) for _ in range(early_legitimate))
    late: list[tuple[str, Any]] = [("fraud", attack) for attack in late_fraud]
    late.extend(
        ("legitimate", None)
        for _ in range(legitimate_count - early_legitimate)
    )
    frozen: list[tuple[str, Any]] = [
        ("fraud", attack) for attack in frozen_fraud
    ]
    simulator = PopulationSimulatorV5(
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
    schema = json.loads(
        (ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8")
    )
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
        "behavior_profile_schema_version": 5 if behavior_profile else None,
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
    profile = None
    if args.behavior_profile:
        payload = json.loads(args.behavior_profile.read_text(encoding="utf-8"))
        profile = BehaviorProfileV5.from_dict(
            payload.get("behavior_profile", payload)
        )
    result = generate_population_dataset_v5(
        args.output,
        events=args.events,
        seed=args.seed,
        fraud_rate=args.fraud_rate,
        benign_adversarial_rate=args.benign_adversarial_rate,
        behavior_profile=profile,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
