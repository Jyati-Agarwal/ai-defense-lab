"""Five-round A+B-only in-distribution hardening arena."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from defend.embeddings import CachedMiniLM
from defend.features import FeatureExtractor
from defend.harness import (
    assert_training_partitions,
    canonical_events_hash,
    make_split_plan,
    model_hash,
    source_tree_hash,
    target,
)
from defend.memory import PrecedentMemory
from defend.models import DefenseModel
from defend.train import (
    ROOT,
    event_provenance,
    load_events_with_provenance,
    observable_copy,
    trusted_registry_from_provenance,
)
from loop.evasion import Candidate, fitness, mutate_training_events


ATTACK_SUCCESS_THRESHOLD = 0.60
REPLAY_CHALLENGE_COUNT = 48


def _json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _plot(path: Path, rounds: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    x = [row["round"] for row in rounds]
    y = [row["asr"] for row in rounds]
    adaptive = [row["adaptive_asr"] for row in rounds]
    axis.plot(x, y, marker="o", linewidth=2.2, color="#ff3b5c", label="Fixed replay ASR")
    axis.plot(x, adaptive, marker="o", linewidth=1.5, linestyle="--", color="#8c94a8", label="Adaptive pressure")
    axis.set(
        xlabel="Arena round",
        ylabel="Attack success rate",
        ylim=(0, 1),
        xticks=x,
        title="Fixed A+B-derived replay challenge — not generalization",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _round_training(plan, generated: list[dict[str, Any]], round_number: int) -> list[dict[str, Any]]:
    base = [event for event in plan.training if event["label"]["split"] == "A"]
    adaptive = [
        event
        for event in plan.training
        if event["label"]["split"] == "B"
        and int(event["label"].get("arena_round") or 1) <= round_number
    ]
    rows = base + adaptive + generated
    assert_training_partitions(rows)
    return sorted(rows, key=lambda event: str(event["ts"]))


def _fused_scores(model: DefenseModel, batch) -> np.ndarray:
    layers = model.layer_scores(batch)
    if model.fusion is None:
        raise RuntimeError("arena model fusion is not fitted")
    return np.maximum(model.fusion.predict_proba(layers), layers[:, 0])


def run_arena(
    events: Iterable[dict[str, Any]],
    results_dir: str | Path = ROOT / "results",
    seed: int = 20260827,
    use_cache: bool = True,
    events_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(events)
    output = Path(results_dir)
    model_path = output / "model.pkl"
    cache_path = output / "arena_log.json"
    plan = make_split_plan(rows)
    input_evidence = {
        "dataset_sha256": canonical_events_hash(rows),
        "split_plan_sha256": plan.digest(),
        "source_tree_sha256": source_tree_hash(ROOT),
        **event_provenance(events_provenance),
    }
    if use_cache and cache_path.is_file() and model_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("seed") == seed and cached.get("final_model_hash") == model_hash(model_path) and all(
            cached.get(key) == value for key, value in input_evidence.items()
        ):
            if not (cached.get("hardening_gate") or {}).get("passed", False):
                raise AssertionError("cached arena replay hardening gate did not pass")
            release_path = output / "arena_release.json"
            if not release_path.is_file():
                raise RuntimeError("arena cache is missing its release binding")
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if (
                release.get("status") != "passed"
                or release.get("seed") != seed
                or release.get("final_model_hash") != cached["final_model_hash"]
                or any(release.get(key) != value for key, value in input_evidence.items())
            ):
                raise RuntimeError("arena release binding does not match the cached final model")
            artifact_hashes = release.get("artifact_sha256") or {}
            if not artifact_hashes or any(
                not (output / name).is_file() or model_hash(output / name) != digest
                for name, digest in artifact_hashes.items()
            ):
                raise RuntimeError("arena release artifact hashes do not match")
            return cached
    manifest_path = output / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("arena requires the training manifest from a fresh train run")
    training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_model_hash = model_hash(model_path)
    if training_manifest.get("model_sha256") != base_model_hash:
        raise RuntimeError("arena requires model.pkl to match the freshly trained base model hash")
    evidence_mismatches = [
        key for key, value in input_evidence.items() if training_manifest.get(key) != value
    ]
    if evidence_mismatches:
        raise RuntimeError(
            "arena training manifest does not match current input evidence: "
            + ", ".join(evidence_mismatches)
        )
    embedder = CachedMiniLM(output / "embeddings.npz")
    memory = PrecedentMemory(embedder, output / "precedent.index", output / "precedent_history.json")
    memory.build(plan.precedent_history)
    extractor = FeatureExtractor(
        embedder,
        memory,
        plan.precedent_history,
        trusted_registry=trusted_registry_from_provenance(events_provenance, rows),
    )
    replay_candidates = mutate_training_events(
        plan.training,
        round_number=0,
        count=REPLAY_CHALLENGE_COUNT,
        seed=seed,
    )
    replay_events = [candidate.event for candidate in replay_candidates]
    replay_batch = extractor.transform(replay_events)
    generated: list[dict[str, Any]] = []
    latest_survivors: list[dict[str, Any]] = []
    reserved_survivors: list[dict[str, Any]] = []
    round_log: list[dict[str, Any]] = []
    phylogeny = [{"id": "root", "label": "Known A training paths", "status": "seed"}]
    final_model: DefenseModel | None = None

    for round_number in range(1, 6):
        training = _round_training(plan, generated, round_number)
        training_batch = extractor.transform(training)
        final_model = DefenseModel(seed=seed + round_number).fit(training_batch, [target(event) for event in training])
        replay_scores = _fused_scores(final_model, replay_batch)
        replay_asr = float(np.mean(replay_scores < ATTACK_SUCCESS_THRESHOLD))
        source_pool = latest_survivors or [event for event in training if target(event)]
        candidates = mutate_training_events(
            source_pool,
            round_number=round_number,
            count=24,
            seed=seed,
        )
        candidate_events = [candidate.event for candidate in candidates]
        candidate_batch = extractor.transform(candidate_events)
        scores = _fused_scores(final_model, candidate_batch)
        payoffs = np.asarray([(event.get("txn") or {}).get("amount_minor", 0) for event in candidate_events], dtype=float)
        ranking = np.argsort([fitness(payoff, score) for payoff, score in zip(payoffs, scores, strict=True)])[::-1]
        actual_positions = list(np.flatnonzero(scores < ATTACK_SUCCESS_THRESHOLD))
        selected_positions = np.asarray(
            list(dict.fromkeys(actual_positions + [int(position) for position in ranking]))[: max(5, len(ranking) // 4)],
            dtype=int,
        )
        latest_survivors = [candidate_events[int(position)] for position in selected_positions]
        for position in actual_positions:
            candidate_events[int(position)]["arena_observation"] = {
                "score_at_discovery": round(float(scores[int(position)]), 8),
                "round": round_number,
                "success_definition": "ALLOW or STEP_UP; HOLD/BLOCK stops payment",
            }
            if round_number == 5:
                reserved_survivors.append(candidate_events[int(position)])
        if round_number < 5:
            generated.extend(latest_survivors)
        adaptive_asr = float(np.mean(scores < ATTACK_SUCCESS_THRESHOLD))
        best = candidates[int(ranking[0])]
        round_log.append(
            {
                "round": round_number,
                "asr": round(replay_asr, 8),
                "adaptive_asr": round(adaptive_asr, 8),
                "mutation": best.mutation,
                "disposition": (
                    "reserved for guarded vaccination holdout"
                    if round_number == 5
                    else "added to guarded A+B training fold"
                ),
                "candidate_count": len(candidates),
                "selected_training_count": 0 if round_number == 5 else len(selected_positions),
                "best_fitness": round(float(fitness(payoffs[int(ranking[0])], scores[int(ranking[0])])), 6),
            }
        )
        displayed_positions = list(dict.fromkeys(actual_positions + [int(position) for position in selected_positions[:3]]))
        for position in displayed_positions:
            candidate = candidates[int(position)]
            phylogeny.append(
                {
                    "id": candidate.event["event_id"],
                    "label": candidate.mutation,
                    "parent": candidate.parent_id if any(node["id"] == candidate.parent_id for node in phylogeny) else "root",
                    "emergent": "+" in candidate.mutation,
                    "status": "survived" if scores[int(position)] < ATTACK_SUCCESS_THRESHOLD else "hard negative",
                }
            )

    final_training = _round_training(plan, generated, 5)
    if final_model is None:
        raise RuntimeError("arena produced no model")
    final_model.history = [observable_copy(event) for event in plan.precedent_history]
    hardening_gate = {
        "metric": "fixed_replay_attack_success_rate",
        "baseline": round_log[0]["asr"],
        "final": round_log[-1]["asr"],
        "operator": "final == 0 or final < baseline",
        "passed": round_log[-1]["asr"] == 0 or round_log[-1]["asr"] < round_log[0]["asr"],
        "outcome": (
            "strict_improvement"
            if round_log[-1]["asr"] < round_log[0]["asr"]
            else "zero_residual_floor"
        ),
    }
    with tempfile.TemporaryDirectory(prefix=".arena-staging-", dir=output) as staging_name:
        staging = Path(staging_name)
        staged_model = staging / "model.pkl"
        joblib.dump(final_model, staged_model, compress=3)
        final_hash = model_hash(staged_model)
        artifact = {
            "seed": seed,
            "label": "Fixed A+B replay security gate — in-distribution; not generalization",
            "rounds": round_log,
            "survivors": sorted({node["label"] for node in phylogeny if node.get("status") == "survived"}),
            "training_partitions": ["A", "B"],
            "input_partitions_materialized": sorted({event["label"]["split"] for event in rows}),
            "partitions_used_for_training_or_feature_extraction": ["A", "B"],
            "success_definition": "fused score below 0.60 (ALLOW or STEP_UP); HOLD/BLOCK stops payment",
            "base_model_hash": base_model_hash,
            "final_training_events": len(final_training),
            "final_model_hash": final_hash,
            "reserved_final_round_survivors": len(reserved_survivors),
            "replay_challenge": {
                "count": len(replay_events),
                "dataset_sha256": canonical_events_hash(replay_events),
                "source_partitions": ["A", "B"],
                "training_included": False,
                "generation": "seeded mutations of fraudulent events from the guarded global training fold",
                "generation_round": 0,
            },
            "hardening_gate": hardening_gate,
            **input_evidence,
        }
        if not hardening_gate["passed"]:
            raise AssertionError(
                f"fixed replay ASR neither improved nor reached zero: {hardening_gate['baseline']:.4f} -> "
                f"{hardening_gate['final']:.4f}"
            )
        staged_survivors = staging / "arena_survivors.jsonl"
        with staged_survivors.open("w", encoding="utf-8") as handle:
            for event in reserved_survivors:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        _json(staging / "arena_log.json", artifact)
        _json(staging / "phylogeny.json", {"nodes": phylogeny})
        _plot(staging / "asr_curve.png", round_log)
        artifact_names = ("model.pkl", "arena_survivors.jsonl", "arena_log.json", "phylogeny.json", "asr_curve.png")
        release = {
            "status": "passed",
            "seed": seed,
            "base_model_hash": base_model_hash,
            "final_model_hash": final_hash,
            "artifact_sha256": {name: model_hash(staging / name) for name in artifact_names},
            **input_evidence,
        }
        _json(staging / "arena_release.json", release)
        for name in artifact_names:
            os.replace(staging / name, output / name)
        os.replace(staging / "arena_release.json", output / "arena_release.json")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    rows, provenance = load_events_with_provenance(args.events)
    run_arena(
        rows,
        args.results,
        args.seed,
        use_cache=not args.rerun,
        events_provenance=provenance,
    )


if __name__ == "__main__":
    main()
