"""Split discipline, static firewall, and frozen-scoring invariants."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TRAINING_PARTITIONS = frozenset({"A", "B"})
FROZEN_PARTITIONS = frozenset({"C", "D", "E"})


def partition_name(event: dict[str, Any]) -> str:
    return str(event["label"]["split"])


def target(event: dict[str, Any]) -> int:
    return int(bool(event["label"]["is_fraud"]))


def model_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_events_hash(events: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_tree_hash(root: str | Path) -> str:
    root_path = Path(root)
    digest = hashlib.sha256()
    paths = (
        sorted((root_path / "defend").rglob("*.py"))
        + sorted((root_path / "loop").rglob("*.py"))
        + [root_path / "contracts" / "event.schema.json"]
    )
    for path in paths:
        digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def generator_input_tree_hash(root: str | Path) -> str:
    root_path = Path(root)
    paths = [
        root_path / "sim" / name
        for name in (
            "__init__.py",
            "population.py",
            "mandate_twin.py",
            "primitives.py",
            "content_llm.py",
        )
    ]
    attack_paths = sorted((root_path / "sim" / "attacks").rglob("*.py"))
    cache_paths = sorted((root_path / "data" / "llm_cache" / "content").glob("*.json"))
    paths += attack_paths + [root_path / "data" / "market_prices.csv"] + cache_paths
    if not attack_paths or not cache_paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError("generator inputs are missing or unsafe")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def assert_training_partitions(events: Iterable[dict[str, Any]]) -> None:
    observed = {partition_name(event) for event in events}
    if not observed <= TRAINING_PARTITIONS:
        raise AssertionError(f"training received evaluation-only partitions: {sorted(observed - TRAINING_PARTITIONS)}")


def import_firewall_violations(root: str | Path) -> list[str]:
    violations: list[str] = []
    for path in Path(root).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(
                name == "importlib"
                or name.startswith("importlib.")
                or name == "sim.attacks"
                or name.startswith("sim.attacks.")
                for name in names
            ) or (
                isinstance(node, ast.Name) and node.id == "__import__"
            ) or (
                isinstance(node, ast.Attribute) and node.attr == "__import__"
            ):
                violations.append(f"{path}:{node.lineno}")
                break
    return violations


@dataclass(frozen=True)
class SplitPlan:
    precedent_history: tuple[dict[str, Any], ...]
    training: tuple[dict[str, Any], ...]
    known: tuple[dict[str, Any], ...]
    adaptive: tuple[dict[str, Any], ...]
    novel: tuple[dict[str, Any], ...]
    heldout: tuple[dict[str, Any], ...]
    benign: tuple[dict[str, Any], ...]
    temporal_cutoff: str

    def evaluation(self) -> tuple[dict[str, Any], ...]:
        return self.known + self.adaptive + self.novel + self.heldout + self.benign

    def digest(self) -> str:
        return canonical_events_hash(
            {"event_id": event["event_id"], "split": event["label"]["split"], "ts": event["ts"]}
            for event in self.precedent_history + self.training + self.evaluation()
        )


def make_split_plan(
    events: Iterable[dict[str, Any]],
    evaluation_fraction: float = 0.25,
    precedent_fraction: float = 0.15,
) -> SplitPlan:
    rows = list(events)
    by_partition = {name: [event for event in rows if partition_name(event) == name] for name in "ABCDE"}
    historical_candidates = [
        event
        for event in sorted(by_partition["A"], key=lambda value: str(value["ts"]))
        if target(event) == 0 and event.get("mandate_chain")
    ]
    history_count = min(
        max(1, len(historical_candidates) // 2),
        max(5, int(len(historical_candidates) * precedent_fraction)),
    )
    history = historical_candidates[:history_count]
    history_ids = {str(event["event_id"]) for event in history}
    candidates = sorted(
        [
            event
            for event in by_partition["A"] + by_partition["B"]
            if str(event["event_id"]) not in history_ids
        ],
        key=lambda event: str(event["ts"]),
    )
    if len(candidates) < 4:
        raise ValueError("A+B needs enough events for a global temporal holdout")
    cut = min(len(candidates) - 1, max(1, int(len(candidates) * (1 - evaluation_fraction))))
    training = candidates[:cut]
    temporal_evaluation = candidates[cut:]
    a_eval = [event for event in temporal_evaluation if partition_name(event) == "A"]
    b_eval = [event for event in temporal_evaluation if partition_name(event) == "B"]
    assert_training_partitions(training)
    if not history:
        raise ValueError("a separate historical legitimate corpus is required for precedent memory")
    if len({target(event) for event in training}) != 2:
        raise ValueError("A+B training fold must contain both classes")
    all_evaluation = temporal_evaluation + by_partition["C"] + by_partition["D"] + by_partition["E"]
    development = history + training
    if not all_evaluation or max(str(event["ts"]) for event in development) >= min(str(event["ts"]) for event in all_evaluation):
        raise ValueError("global temporal split violated: every training event must precede every evaluation event")
    return SplitPlan(
        precedent_history=tuple(history),
        training=tuple(training),
        known=tuple(a_eval),
        adaptive=tuple(b_eval),
        novel=tuple(by_partition["C"]),
        heldout=tuple(by_partition["D"]),
        benign=tuple(by_partition["E"]),
        temporal_cutoff=min(str(event["ts"]) for event in all_evaluation),
    )


class FrozenPartitionScorer:
    """Allows each evaluation-only partition one invocation for one frozen model."""

    def __init__(self) -> None:
        self.scored: set[str] = set()

    def claim(self, name: str) -> None:
        if name not in FROZEN_PARTITIONS:
            raise ValueError(f"{name} is not a frozen evaluation partition")
        if name in self.scored:
            raise RuntimeError(f"partition {name} was already scored")
        self.scored.add(name)
