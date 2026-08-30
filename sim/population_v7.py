"""Schema-7 population corrections over the immutable V6 renderer."""

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
    STRATUM_CELL_COUNTS,
    STRATUM_CELL_SHAPES,
    STRATUM_PRIMITIVES,
    BehaviorProfileV6,
    CheckerboardStratumV6,
    PopulationSimulatorV6,
    moment_matched_hour_counts,
)


REPEAT_INVOICE_COUNT_MAX = 256


@dataclass(frozen=True)
class BehaviorProfileV7(BehaviorProfileV6):
    """V6 profile plus the exact calibration repeat-count marginal."""

    repeat_invoice_count_pmf: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            len(self.repeat_invoice_count_pmf) != REPEAT_INVOICE_COUNT_MAX + 1
            or any(
                not math.isfinite(value) or value < 0
                for value in self.repeat_invoice_count_pmf
            )
            or self.repeat_invoice_count_pmf[0] != 0
            or self.repeat_invoice_count_pmf[1] != 0
            or not math.isclose(sum(self.repeat_invoice_count_pmf), 1.0, abs_tol=1e-12)
        ):
            raise ValueError("v7 repeat invoice-count PMF is invalid")

    def as_dict(self) -> dict[str, Any]:
        value = super().as_dict()
        value["schema_version"] = 7
        value["repeat_invoice_count_pmf"] = list(self.repeat_invoice_count_pmf)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorProfileV7":
        if set(value) != {
            "schema_version",
            "source",
            "checkerboard",
            "same_day_repeat_atom",
            "renderer",
            "empirical_bayes",
            "privacy",
            "repeat_invoice_count_pmf",
        } or value.get("schema_version") != 7:
            raise ValueError("only the exact schema-7 behavior profile is accepted")
        inherited = dict(value)
        probabilities = inherited.pop("repeat_invoice_count_pmf")
        inherited["schema_version"] = 6
        base = BehaviorProfileV6.from_dict(inherited)
        return cls(
            **base.__dict__,
            repeat_invoice_count_pmf=tuple(map(float, probabilities)),
        )

    def _sample_repeats(self, count: int, rng: Any) -> list[dict[str, float]]:
        import numpy as np

        rows = super()._sample_repeats(count, rng)
        if not rows:
            return rows
        allocated = _residual_cell_counts(self.repeat_invoice_count_pmf, count, rng)
        values = np.repeat(np.arange(REPEAT_INVOICE_COUNT_MAX + 1), allocated)
        order = sorted(
            range(count), key=lambda index: (rows[index]["invoice_count"], index)
        )
        for index, value in zip(order, values, strict=True):
            rows[index]["invoice_count"] = float(value)
        if any(row["invoice_count"] < 2 for row in rows):
            raise RuntimeError("v7 repeat PMF left repeat support")
        return rows


class PopulationSimulatorV7(PopulationSimulatorV6):
    """V6 renderer with deterministic balanced multiday session anchors."""

    behavior_profile: BehaviorProfileV7

    def _session_days(
        self, first_day: int, last_day: int, sessions: int
    ) -> list[int]:
        if sessions == 1:
            return [first_day]
        m = sessions - 1
        span = last_day - first_day
        if sessions < 2 or span < 2 * m:
            raise ValueError("v7 multiday anchors require existing two-day feasibility")
        days = [first_day + (index * span + m // 2) // m for index in range(sessions)]
        if (
            days[0] != first_day
            or days[-1] != last_day
            or any(right - left < 2 for left, right in zip(days, days[1:]))
        ):
            raise RuntimeError("v7 balanced session anchors violated their support")
        return days


def generate_population_dataset_v7(
    output: Path,
    *,
    events: int = 50_000,
    seed: int = 20260827,
    fraud_rate: float = 0.0055,
    benign_adversarial_rate: float = 0.02,
    behavior_profile: BehaviorProfileV7 | None = None,
) -> dict[str, Any]:
    """Generate the canonical exact-size event file through the V7 scheduler."""
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
    simulator = PopulationSimulatorV7(
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
            raise RuntimeError("v7 simulator clock regressed during canonical generation")
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
                raise RuntimeError("v7 canonical output timestamps are not monotonic")
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
        "behavior_profile_schema_version": 7 if behavior_profile else None,
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
        profile = BehaviorProfileV7.from_dict(payload.get("behavior_profile", payload))
    print(json.dumps(generate_population_dataset_v7(args.output, events=args.events, seed=args.seed, fraud_rate=args.fraud_rate, benign_adversarial_rate=args.benign_adversarial_rate, behavior_profile=profile), sort_keys=True))


if __name__ == "__main__":
    main()
