"""Validate the atlas and render its deterministic UI matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "results" / "atlas_matrix.json"
STAGES = (
    "reconnaissance",
    "context_manip",
    "intent_manip",
    "tool_manip",
    "merchant_manip",
    "payment_manip",
    "delayed_exec",
    "cash_out",
)


def spec_paths(root: Path = ROOT) -> list[Path]:
    return sorted((*root.glob("classic/GPF-*.yaml"), *root.glob("agentic/GPF-*.yaml")))


def load_and_validate(root: Path = ROOT) -> list[dict[str, Any]]:
    schema = yaml.safe_load((root / "schema.yaml").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    attacks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    for path in spec_paths(root):
        attack = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(attack), key=lambda error: list(error.path))
        if errors:
            detail = "; ".join(error.message for error in errors)
            raise ValueError(f"{path}: {detail}")
        if path.stem != attack["id"]:
            raise ValueError(f"{path}: filename must match id {attack['id']}")
        expected_tier = "classic" if path.parent.name == "classic" else "agentic"
        if attack["tier"] != expected_tier:
            raise ValueError(f"{path}: tier must be {expected_tier}")
        positions = [STAGES.index(stage) for stage in attack["primitives"]]
        if positions != sorted(positions):
            raise ValueError(f"{path}: primitives must follow lifecycle order")
        if attack["id"] in seen_ids or attack["name"] in seen_names:
            raise ValueError(f"{path}: duplicate id or name")
        seen_ids.add(attack["id"])
        seen_names.add(attack["name"])
        attacks.append(attack)

    counts = Counter(attack["tier"] for attack in attacks)
    if counts != {"classic": 20, "agentic": 28}:
        raise ValueError(f"atlas requires 20 classic and 28 agentic attacks, got {dict(counts)}")
    return sorted(attacks, key=lambda attack: attack["id"])


def build_matrix(attacks: list[dict[str, Any]]) -> dict[str, Any]:
    columns = []
    for stage in STAGES:
        members = [attack["id"] for attack in attacks if attack["kill_chain_stage"] == stage]
        if not members:
            raise ValueError(f"kill-chain stage {stage} is empty")
        columns.append({"stage": stage, "attack_ids": members})
    return {
        "schema_version": 1,
        "total": len(attacks),
        "tier_counts": dict(Counter(attack["tier"] for attack in attacks)),
        "kill_chain": columns,
        "attacks": attacks,
    }


def render(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    matrix = build_matrix(load_and_validate())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    matrix = build_matrix(load_and_validate())
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"attacks": matrix["total"], "output": None if args.check_only else str(args.output)}))


if __name__ == "__main__":
    main()
