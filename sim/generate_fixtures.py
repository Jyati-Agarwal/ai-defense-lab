"""Generate the checked-in, API-free downstream event fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from sim.attacks import all_attacks, apply_attack
from sim.mandate_twin import DETERMINISTIC_SIGNING_AVAILABLE
from sim.population import PopulationSimulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "fixtures" / "events.jsonl"


def generate_fixture(
    output: Path = DEFAULT_OUTPUT,
    *,
    seed: int = 20260827,
    legitimate: int = 220,
    weird_legitimate: int = 20,
    per_attack: int = 5,
) -> dict[str, object]:
    if not DETERMINISTIC_SIGNING_AVAILABLE:
        raise RuntimeError("deterministic fixtures require cryptography >= 43")
    simulator = PopulationSimulator(seed=seed)
    attacks = all_attacks(include_heldout=True)
    training_attacks = [attack for attack in attacks if attack.split in {"A", "B"}]
    frozen_attacks = [attack for attack in attacks if attack.split in {"C", "D"}]
    early_legitimate = round(legitimate * 0.75)
    early_attack_count = max(1, round(per_attack * 0.75))
    early_schedule = [("legitimate", None)] * early_legitimate + [
        ("attack", attack) for attack in training_attacks for _ in range(early_attack_count)
    ]
    late_schedule = [("legitimate", None)] * (legitimate - early_legitimate) + [
        ("attack", attack) for attack in training_attacks for _ in range(per_attack - early_attack_count)
    ]
    simulator.rng.shuffle(early_schedule)
    simulator.rng.shuffle(late_schedule)

    events = []
    for kind, attack in early_schedule + late_schedule:
        if kind == "legitimate":
            events.append(simulator.generate_legitimate())
        else:
            base = simulator.generate_legitimate(force_channel=attack.base_channel)
            event = apply_attack(base, attack, registry=simulator.registry)
            simulator.synchronize_clock(event)
            events.append(event)
    for attack in frozen_attacks:
        for _ in range(per_attack):
            base = simulator.generate_legitimate(force_channel=attack.base_channel)
            event = apply_attack(base, attack, registry=simulator.registry)
            simulator.synchronize_clock(event)
            events.append(event)
    events.extend(simulator.generate_legitimate(weird=True) for _ in range(weird_legitimate))

    schema = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for event in events:
        validator.validate(event)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events)
    output.write_text(payload, encoding="utf-8")
    labels = Counter(event["label"]["split"] for event in events)
    families = Counter(event["label"]["attack_family"] for event in events if event["label"]["is_fraud"])
    manifest = {
        "schema_version": 1,
        "purpose": "deterministic demo fixture; not a production base-rate sample",
        "seed": seed,
        "events": len(events),
        "fraud_events": sum(event["label"]["is_fraud"] for event in events),
        "split_counts": dict(sorted(labels.items())),
        "attack_family_counts": dict(sorted(families.items())),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "synthetic_pii": True,
        "api_calls": 0,
        **simulator.registry.manifest_fields(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--legitimate", type=int, default=220)
    parser.add_argument("--weird-legitimate", type=int, default=20)
    parser.add_argument("--per-attack", type=int, default=5)
    args = parser.parse_args()
    manifest = generate_fixture(
        args.output,
        seed=args.seed,
        legitimate=args.legitimate,
        weird_legitimate=args.weird_legitimate,
        per_attack=args.per_attack,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
