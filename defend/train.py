"""Train the frozen Blue model from A+B only.

The complete release pipeline is orchestrated by ``make all``; this module
deliberately performs only the training stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from defend.embeddings import CachedMiniLM
from defend.features import FeatureBatch, FeatureExtractor
from defend.harness import (
    SplitPlan,
    assert_training_partitions,
    canonical_events_hash,
    make_split_plan,
    model_hash,
    source_tree_hash,
    target,
)
from defend.memory import PrecedentMemory
from defend.models import DefenseModel
from sim.mandate_twin import (
    SYNTHETIC_IDENTITY_SEED,
    SYNTHETIC_KEY_SCOPE,
    key_registry_sha256,
    validate_synthetic_event_registry,
    validate_synthetic_key_registry,
)


ROOT = Path(__file__).resolve().parents[1]
EVENT_PROVENANCE_FIELDS = (
    "events_file_sha256",
    "events_manifest_sha256",
    "behavior_profile_sha256",
    "evaluation_preregistration_sha256",
    "key_registry_sha256",
)


def load_events(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} is missing; run `make all` from the repository root"
        )
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return list(payload.get("events", payload) if isinstance(payload, dict) else payload)
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_provenance(
    provenance: Mapping[str, Any] | None,
) -> dict[str, str | None]:
    """Return the stable event-file evidence carried between release stages."""

    if provenance is None:
        return {key: None for key in EVENT_PROVENANCE_FIELDS}
    missing = [key for key in EVENT_PROVENANCE_FIELDS if key not in provenance]
    if missing:
        raise ValueError("event provenance is missing required fields: " + ", ".join(missing))
    return {key: provenance[key] for key in EVENT_PROVENANCE_FIELDS}


def load_events_with_provenance(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a CLI event file and validate its adjacent generation manifest."""

    source = Path(path)
    rows = load_events(source)
    manifest_path = source.with_name(f"{source.stem}.manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} is missing; CLI event inputs require an adjacent manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"{manifest_path} is not a readable event manifest") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{manifest_path} must contain a JSON object")

    actual_file_hash = model_hash(source)
    declared_file_hash = manifest.get("sha256")
    if declared_file_hash != actual_file_hash:
        raise RuntimeError(
            "event manifest SHA mismatch: "
            f"declared {declared_file_hash!r}, actual {actual_file_hash}"
        )
    declared_count = manifest.get("events")
    if declared_count != len(rows):
        raise RuntimeError(
            "event manifest count mismatch: "
            f"declared {declared_count!r}, actual {len(rows)}"
        )

    behavior_profile_hash = manifest.get("behavior_profile_sha256")
    if behavior_profile_hash is not None and not (
        isinstance(behavior_profile_hash, str)
        and len(behavior_profile_hash) == 64
        and all(character in "0123456789abcdef" for character in behavior_profile_hash.lower())
    ):
        raise RuntimeError("event manifest behavior_profile_sha256 must be a SHA-256 hex digest or null")
    evaluation_preregistration_hash = manifest.get("evaluation_preregistration_sha256")
    if evaluation_preregistration_hash is not None and not (
        isinstance(evaluation_preregistration_hash, str)
        and len(evaluation_preregistration_hash) == 64
        and all(
            character in "0123456789abcdef"
            for character in evaluation_preregistration_hash.lower()
        )
    ):
        raise RuntimeError(
            "event manifest evaluation_preregistration_sha256 must be a SHA-256 hex digest or null"
        )
    if manifest.get("key_registry_scope") != SYNTHETIC_KEY_SCOPE:
        raise RuntimeError("event manifest key_registry_scope is invalid or missing")
    try:
        key_registry = validate_synthetic_event_registry(
            rows,
            manifest.get("key_registry"),
            manifest.get("synthetic_identity_seed"),
        )
    except ValueError as error:
        raise RuntimeError(f"event manifest key_registry is invalid: {error}") from error
    if not key_registry:
        raise RuntimeError("event manifest key_registry must not be empty")
    registry_hash = key_registry_sha256(key_registry)
    if manifest.get("key_registry_sha256") != registry_hash:
        raise RuntimeError("event manifest key_registry_sha256 does not match key_registry")
    return rows, {
        "events_file_sha256": actual_file_hash,
        "events_manifest_sha256": model_hash(manifest_path),
        "behavior_profile_sha256": behavior_profile_hash,
        "evaluation_preregistration_sha256": evaluation_preregistration_hash,
        "key_registry_sha256": registry_hash,
        "key_registry": key_registry,
    }


def trusted_registry_from_provenance(
    provenance: Mapping[str, Any] | None,
    events: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    if provenance is None:
        return {}
    try:
        registry = validate_synthetic_key_registry(
            provenance.get("key_registry"), SYNTHETIC_IDENTITY_SEED
        )
    except ValueError as error:
        raise RuntimeError("trusted key registry is invalid or missing from event provenance") from error
    if not registry:
        raise RuntimeError("trusted key registry is empty")
    if provenance.get("key_registry_sha256") != key_registry_sha256(registry):
        raise RuntimeError("trusted key registry does not match event provenance")
    try:
        return validate_synthetic_event_registry(
            events, registry, SYNTHETIC_IDENTITY_SEED
        )
    except ValueError as error:
        raise RuntimeError("trusted key registry does not exactly cover event signers") from error


def observable_copy(event: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(event)
    row.pop("label", None)
    row.pop("scores", None)
    row.pop("decision", None)
    return row


@dataclass
class TrainingRun:
    model: DefenseModel
    plan: SplitPlan
    batch: FeatureBatch
    model_path: Path


def train_events(
    events: Iterable[dict[str, Any]],
    results_dir: str | Path = ROOT / "results",
    seed: int = 20260827,
    events_provenance: Mapping[str, Any] | None = None,
) -> TrainingRun:
    rows = list(events)
    plan = make_split_plan(rows)
    assert_training_partitions(plan.training)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    embedder = CachedMiniLM(output / "embeddings.npz")
    memory = PrecedentMemory(embedder, output / "precedent.index", output / "precedent_history.json")
    memory.build(plan.precedent_history)
    embedder.encode(
        [
            "Find running shoes under INR 5000",
            "Digital wallet voucher",
            "Trail running shoes",
            "running shoes",
            "wallet",
        ]
    )
    embedder.flush()
    extractor = FeatureExtractor(
        embedder,
        memory,
        plan.precedent_history,
        trusted_registry=trusted_registry_from_provenance(events_provenance, rows),
    )
    batch = extractor.transform(plan.training)
    targets = [target(event) for event in plan.training]
    model = DefenseModel(seed=seed).fit(batch, targets)
    model.history = [observable_copy(event) for event in plan.precedent_history]
    model_path = output / "model.pkl"
    temporary = output / "model.tmp.pkl"
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, model_path)
    manifest = {
        "seed": seed,
        "training_partitions": ["A", "B"],
        "temporal_split": "single global chronological cutoff; every training event precedes every evaluation event",
        "temporal_cutoff": plan.temporal_cutoff,
        "training_events": len(plan.training),
        "precedent_history_events": len(plan.precedent_history),
        "evaluation_counts": {
            "A": len(plan.known),
            "B": len(plan.adaptive),
            "C": len(plan.novel),
            "D": len(plan.heldout),
            "E": len(plan.benign),
        },
        "lightgbm_cv_folds": model.tree.cv_folds_ if model.tree else 0,
        "isotonic_calibration": True,
        "fusion": "five-layer logistic regression",
        "embedding_model": embedder.model_name,
        "precedent_self_matches": False,
        "dataset_sha256": canonical_events_hash(rows),
        "split_plan_sha256": plan.digest(),
        "source_tree_sha256": source_tree_hash(ROOT),
        **event_provenance(events_provenance),
        "model_sha256": model_hash(model_path),
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return TrainingRun(model, plan, batch, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--all",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.all:
        parser.error("--all was removed; use `make all` as the single release orchestrator")
    rows, provenance = load_events_with_provenance(args.events)
    train_events(rows, args.results, args.seed, events_provenance=provenance)


if __name__ == "__main__":
    main()
