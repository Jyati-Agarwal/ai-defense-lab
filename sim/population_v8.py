"""Schema-8 mixed-resolution checkerboard over the immutable V7 renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sim.population import ROOT
from sim.population_v4 import _residual_cell_counts
from sim.population_v6 import (
    ATOM_SELECTION_NAMESPACE,
    ATOM_SELECTION_RULE,
    CHECKERBOARD_BINS,
    CHECKERBOARD_STRATA,
    LOCALIZATION_FACTOR_DEFAULT,
    MIN_CHECKERBOARD_CUSTOMERS,
    QUANTILE_PROBABILITIES,
    REPEAT_PRIMITIVES,
    SAMPLED_PRIMITIVE_FIELDS,
    SINGLETON_CONTINUOUS_PRIMITIVES,
    STRATUM_CELL_SHAPES as V6_STRATUM_CELL_SHAPES,
    STRATUM_PRIMITIVES,
    CheckerboardStratumV6,
    moment_matched_hour_counts,
    _require_keys,
)
from sim.population_v7 import (
    REPEAT_INVOICE_COUNT_MAX,
    BehaviorProfileV7,
    PopulationSimulatorV7,
)


REPEAT_CHECKERBOARD_SHAPE = (32,) + (CHECKERBOARD_BINS,) * 7
STRATUM_CELL_SHAPES = {
    "singleton": V6_STRATUM_CELL_SHAPES["singleton"],
    "multiday_repeat": REPEAT_CHECKERBOARD_SHAPE,
}
STRATUM_CELL_COUNTS = {
    name: math.prod(shape) for name, shape in STRATUM_CELL_SHAPES.items()
}


@dataclass(frozen=True)
class CheckerboardStratumV8(CheckerboardStratumV6):
    """V6 singleton grid or the V8 32x3^7 multiday grid."""

    def __post_init__(self) -> None:
        import numpy as np

        if self.name == "singleton":
            super().__post_init__()
            return
        if self.name != "multiday_repeat":
            raise ValueError("unknown v8 checkerboard stratum")
        if self.customer_count < MIN_CHECKERBOARD_CUSTOMERS:
            raise ValueError("every v8 checkerboard needs at least 32 customers")
        if (
            len(self.primitive_quantiles) != len(REPEAT_PRIMITIVES)
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
            raise ValueError("v8 checkerboard quantiles are invalid")
        quantiles = dict(zip(REPEAT_PRIMITIVES, self.primitive_quantiles, strict=True))
        if (
            any(
                value < 2 or not float(value).is_integer()
                for value in quantiles["invoice_count"]
            )
            or any(
                not 0 <= value < 1
                for name in ("active_end_fraction", "span_within_end_fraction")
                for value in quantiles[name]
            )
            or any(value <= 0 for value in quantiles["eb_log_amount_scale"])
            or any(
                not 0 <= value <= 1
                for name in ("eb_short_session_probability", "circadian_resultant")
                for value in quantiles[name]
            )
            or any(
                not 0 <= value < 2 * math.pi
                for value in quantiles["circadian_phase_latent"]
            )
        ):
            raise ValueError("v8 repeat quantiles left their support")
        masses = np.asarray(self.cell_masses, dtype=float)
        if (
            masses.shape != (STRATUM_CELL_COUNTS[self.name],)
            or not np.isfinite(masses).all()
            or (masses < 0).any()
            or not math.isclose(float(masses.sum()), 1.0, abs_tol=2e-10)
        ):
            raise ValueError("v8 checkerboard masses are invalid")
        tensor = masses.reshape(REPEAT_CHECKERBOARD_SHAPE)
        for coordinate, bins in enumerate(REPEAT_CHECKERBOARD_SHAPE):
            axes = tuple(index for index in range(len(REPEAT_CHECKERBOARD_SHAPE)) if index != coordinate)
            marginal = tensor.sum(axis=axes)
            if any(
                not math.isclose(float(value), 1 / bins, abs_tol=2e-10)
                for value in marginal
            ):
                raise ValueError("v8 checkerboard marginals differ from their fixed axis bins")

    def as_dict(self) -> dict[str, Any]:
        return {
            "customers": self.customer_count,
            "primitives": list(self.primitives),
            "cell_shape": list(STRATUM_CELL_SHAPES[self.name]),
            "primitive_quantiles": {
                name: list(values)
                for name, values in zip(
                    SINGLETON_CONTINUOUS_PRIMITIVES
                    if self.name == "singleton"
                    else REPEAT_PRIMITIVES,
                    self.primitive_quantiles,
                    strict=True,
                )
            },
            "cell_masses": list(self.cell_masses),
        }

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "CheckerboardStratumV8":
        _require_keys(
            value,
            {"customers", "primitives", "cell_shape", "primitive_quantiles", "cell_masses"},
            f"v8 {name} stratum",
        )
        continuous = (
            SINGLETON_CONTINUOUS_PRIMITIVES if name == "singleton" else REPEAT_PRIMITIVES
        )
        if (
            name not in CHECKERBOARD_STRATA
            or tuple(value["primitives"]) != STRATUM_PRIMITIVES[name]
            or tuple(value["cell_shape"]) != STRATUM_CELL_SHAPES[name]
            or set(value["primitive_quantiles"]) != set(continuous)
        ):
            raise ValueError("v8 checkerboard structure changed")
        return cls(
            name=name,
            customer_count=int(value["customers"]),
            primitive_quantiles=tuple(
                tuple(map(float, value["primitive_quantiles"][primitive]))
                for primitive in continuous
            ),
            cell_masses=tuple(map(float, value["cell_masses"])),
        )


@dataclass(frozen=True)
class BehaviorProfileV8(BehaviorProfileV7):
    """V7 profile with the V8 mixed-resolution multiday checkerboard."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all(isinstance(stratum, CheckerboardStratumV8) for stratum in self.strata):
            raise ValueError("v8 profile requires v8 checkerboard strata")

    def as_dict(self) -> dict[str, Any]:
        value = super().as_dict()
        value["schema_version"] = 8
        value["checkerboard"]["repeat_count_bins"] = REPEAT_CHECKERBOARD_SHAPE[0]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorProfileV8":
        if set(value) != {
            "schema_version",
            "source",
            "checkerboard",
            "same_day_repeat_atom",
            "renderer",
            "empirical_bayes",
            "privacy",
            "repeat_invoice_count_pmf",
        } or value.get("schema_version") != 8:
            raise ValueError("only the exact schema-8 behavior profile is accepted")
        source = value["source"]
        checkerboard = value["checkerboard"]
        atom = value["same_day_repeat_atom"]
        renderer = value["renderer"]
        eb = value["empirical_bayes"]
        _require_keys(source, {"dataset", "partition", "customer_ids_sha256", "transactions_sha256", "customers", "transactions"}, "v8 source")
        _require_keys(checkerboard, {"bins", "repeat_count_bins", "quantile_probabilities", "strata", "smoothing", "pseudocounts", "stored_rank_rows"}, "v8 checkerboard")
        _require_keys(atom, {"condition", "repeat_customers", "probability", "allocation", "selection", "projection"}, "v8 atom")
        _require_keys(renderer, {"hour_weights", "calendar_weights", "observation_days", "session_gap_log_mean", "session_gap_log_std", "circadian_phase_cut", "global_log_amount_center", "source_minor_units_per_major", "localization_factor"}, "v8 renderer")
        _require_keys(eb, {"amount_variance_prior_alpha", "amount_variance_prior_beta", "short_session_prior_alpha", "short_session_prior_beta"}, "v8 empirical-Bayes")
        if (
            checkerboard["bins"] != CHECKERBOARD_BINS
            or checkerboard["repeat_count_bins"] != REPEAT_CHECKERBOARD_SHAPE[0]
            or tuple(checkerboard["quantile_probabilities"]) != QUANTILE_PROBABILITIES
            or set(checkerboard["strata"]) != set(CHECKERBOARD_STRATA)
            or checkerboard["smoothing"] is not False
            or checkerboard["pseudocounts"] is not False
            or checkerboard["stored_rank_rows"] is not False
            or atom.get("condition") != "invoice_count>=2"
            or atom.get("allocation") != "exact round(probability*sampled_repeat_count)"
            or atom.get("selection") != ATOM_SELECTION_RULE
            or atom.get("projection") != "retain sampled end day and set first day equal to last day"
            or renderer.get("source_minor_units_per_major") != 100
            or value["privacy"] != {
                "fixed_size_aggregate": True,
                "contains_customer_ids": False,
                "contains_raw_rows": False,
                "contains_rank_rows": False,
                "contains_evaluator_feature_rows": False,
                "contains_prediction_vectors": False,
                "differential_privacy_claimed": False,
            }
        ):
            raise ValueError("v8 structural contract changed")
        return cls(
            source_dataset=str(source["dataset"]),
            source_partition=str(source["partition"]),
            source_customer_ids_sha256=str(source["customer_ids_sha256"]),
            source_transactions_sha256=str(source["transactions_sha256"]),
            source_customer_count=int(source["customers"]),
            source_transaction_count=int(source["transactions"]),
            repeat_customer_count=int(atom["repeat_customers"]),
            same_day_repeat_probability=float(atom["probability"]),
            quantile_probabilities=tuple(map(float, checkerboard["quantile_probabilities"])),
            strata=tuple(
                CheckerboardStratumV8.from_dict(name, checkerboard["strata"][name])
                for name in CHECKERBOARD_STRATA
            ),
            hour_weights=tuple(map(float, renderer["hour_weights"])),
            calendar_weights=tuple(map(float, renderer["calendar_weights"])),
            observation_days=int(renderer["observation_days"]),
            session_gap_log_mean=float(renderer["session_gap_log_mean"]),
            session_gap_log_std=float(renderer["session_gap_log_std"]),
            circadian_phase_cut=float(renderer["circadian_phase_cut"]),
            global_log_amount_center=float(renderer["global_log_amount_center"]),
            localization_factor=float(renderer["localization_factor"]),
            amount_variance_prior_alpha=float(eb["amount_variance_prior_alpha"]),
            amount_variance_prior_beta=float(eb["amount_variance_prior_beta"]),
            short_session_prior_alpha=float(eb["short_session_prior_alpha"]),
            short_session_prior_beta=float(eb["short_session_prior_beta"]),
            repeat_invoice_count_pmf=tuple(map(float, value["repeat_invoice_count_pmf"])),
        )

    def _sample_repeats(self, count: int, rng: Any) -> list[dict[str, float]]:
        import numpy as np

        if count == 0:
            return []
        stratum = self.strata[1]
        cell_counts = _residual_cell_counts(stratum.cell_masses, count, rng)
        indices = np.repeat(np.arange(len(cell_counts)), cell_counts)
        rng.shuffle(indices)
        cells = np.column_stack(np.unravel_index(indices, REPEAT_CHECKERBOARD_SHAPE))
        uniforms = (cells + rng.random((count, len(REPEAT_PRIMITIVES)))) / np.asarray(
            REPEAT_CHECKERBOARD_SHAPE
        )
        probabilities = np.asarray(self.quantile_probabilities, dtype=float)
        sampled = np.empty((count, len(REPEAT_PRIMITIVES)), dtype=float)
        for index, quantiles in enumerate(stratum.primitive_quantiles[1:], start=1):
            sampled[:, index] = np.interp(uniforms[:, index], probabilities, quantiles)

        allocated_counts = _residual_cell_counts(
            self.repeat_invoice_count_pmf, count, rng
        )
        invoice_counts = np.repeat(
            np.arange(REPEAT_INVOICE_COUNT_MAX + 1), allocated_counts
        )
        count_order = sorted(range(count), key=lambda index: (uniforms[index, 0], index))
        sampled[count_order, 0] = invoice_counts
        expected_histogram = {
            value: int(allocated_counts[value])
            for value in range(2, REPEAT_INVOICE_COUNT_MAX + 1)
            if allocated_counts[value]
        }
        if (
            any(value < 2 for value in sampled[:, 0])
            or Counter(map(int, sampled[:, 0])) != expected_histogram
        ):
            raise RuntimeError("v8 exact repeat PMF allocation changed")

        output = []
        for values in sampled:
            row = dict(zip(REPEAT_PRIMITIVES, map(float, values), strict=True))
            u_end = row["active_end_fraction"]
            u_span = row["span_within_end_fraction"]
            if (
                not 0 <= u_end < 1
                or not 0 <= u_span < 1
                or row["eb_log_amount_scale"] <= 0
                or not 0 <= row["eb_short_session_probability"] <= 1
                or not 0 <= row["circadian_resultant"] <= 1
            ):
                raise RuntimeError("v8 repeat primitive left its fitted support")
            last_day = 1 + math.floor(u_end * (self.observation_days - 1))
            span_days = 1 + math.floor(u_span * last_day)
            first_day = last_day - span_days
            if not 0 <= first_day < last_day <= self.observation_days - 1:
                raise RuntimeError("v8 decoded repeat days are invalid")
            phase = (
                row["circadian_phase_latent"] + self.circadian_phase_cut
            ) % (2 * math.pi)
            row.update(
                {
                    "active_day_position": 0.0,
                    "circadian_phase_sin": math.sin(phase),
                    "circadian_phase_cos": math.cos(phase),
                    "eb_circadian_concentration": 0.0,
                    "fixed_hour": -1.0,
                    "first_day": float(first_day),
                    "last_day": float(last_day),
                    "active_start_fraction": first_day / self.observation_days,
                    "active_duration_fraction": (last_day - first_day)
                    / self.observation_days,
                    "same_day_atom": 0.0,
                }
            )
            output.append(row)
        return output


class PopulationSimulatorV8(PopulationSimulatorV7):
    """The V7 renderer driven by V8 checkerboard samples."""

    behavior_profile: BehaviorProfileV8


def generate_population_dataset_v8(
    output: Path,
    *,
    events: int = 50_000,
    seed: int = 20260827,
    fraud_rate: float = 0.0055,
    benign_adversarial_rate: float = 0.02,
    behavior_profile: BehaviorProfileV8 | None = None,
) -> dict[str, Any]:
    """Generate the canonical exact-size event file through the V8 scheduler."""
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
        else:
            cut = max(1, round(len(occurrences) * 0.75))
            early_fraud.extend(occurrences[:cut])
            late_fraud.extend(occurrences[cut:])
    legitimate_count = events - fraud_count - benign_count
    early_legitimate = round(legitimate_count * 0.75)
    early = [("fraud", attack) for attack in early_fraud] + [("legitimate", None)] * early_legitimate
    late = [("fraud", attack) for attack in late_fraud] + [("legitimate", None)] * (legitimate_count - early_legitimate)
    frozen = [("fraud", attack) for attack in frozen_fraud]
    simulator = PopulationSimulatorV8(
        seed=seed,
        customers=max(1, round(events / behavior_profile.activity_mean_transactions)) if behavior_profile else 800,
        behavior_profile=behavior_profile,
    )
    simulator._prepare_activity_quotas(events)
    for phase in (early, late, frozen):
        simulator.rng.shuffle(phase)
    schedule = early + late + frozen + [("E", None)] * benign_count
    schema = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    generated_events: list[dict[str, Any]] = []
    previous_clock = simulator.clock
    for kind, attack in schedule:
        if kind == "fraud":
            event = apply_attack(
                simulator.generate_legitimate(force_channel=attack.base_channel),
                attack,
                registry=simulator.registry,
            )
        elif kind == "E":
            event = simulator.generate_legitimate(weird=True)
        else:
            event = simulator.generate_legitimate()
        if simulator.clock < previous_clock:
            raise RuntimeError("v8 simulator clock regressed during canonical generation")
        previous_clock = simulator.clock
        generated_events.append(event)
    generated_events.sort(
        key=lambda event: datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00"))
    )
    with temporary.open("w", encoding="utf-8") as handle:
        previous_timestamp: datetime | None = None
        for event in generated_events:
            timestamp = datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00"))
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise RuntimeError("v8 canonical output timestamps are not monotonic")
            previous_timestamp = timestamp
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
        "behavior_profile_schema_version": 8 if behavior_profile else None,
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
        profile = BehaviorProfileV8.from_dict(payload.get("behavior_profile", payload))
    print(json.dumps(generate_population_dataset_v8(args.output, events=args.events, seed=args.seed, fraud_rate=args.fraud_rate, benign_adversarial_rate=args.benign_adversarial_rate, behavior_profile=profile), sort_keys=True))


if __name__ == "__main__":
    main()
