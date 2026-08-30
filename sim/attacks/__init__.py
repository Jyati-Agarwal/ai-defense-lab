"""Concrete attacks assembled from lifecycle primitives."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

from sim.attacks.catalog import ATTACKS
from sim.primitives import AttackDefinition, apply_definition


def all_attacks(*, include_heldout: bool = False) -> list[AttackDefinition]:
    attacks = list(ATTACKS)
    if include_heldout:
        for module_name in (
            "sim.attacks.heldout.ttl_dormancy",
            "sim.attacks.heldout.bust_out",
        ):
            attacks.append(import_module(module_name).ATTACK)
    return attacks


def attacks_for_split(
    splits: Iterable[str],
    *,
    include_heldout: bool = False,
) -> list[AttackDefinition]:
    selected = set(splits)
    return [attack for attack in all_attacks(include_heldout=include_heldout) if attack.split in selected]


apply_attack = apply_definition


__all__ = ["AttackDefinition", "all_attacks", "apply_attack", "attacks_for_split"]
