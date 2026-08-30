"""Guarded vaccination candidate with a reserved survivor holdout and human boundary."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.tree import DecisionTreeClassifier, export_text

from defend.embeddings import CachedMiniLM
from defend.features import FEATURE_COLUMNS, FeatureExtractor
from defend.harness import canonical_events_hash, make_split_plan, model_hash, target
from defend.memory import PrecedentMemory
from defend.models import DefenseModel
from defend.train import (
    ROOT,
    event_provenance,
    load_events_with_provenance,
    trusted_registry_from_provenance,
)


STOP_THRESHOLD = 0.60
PATCH_SCORE = 0.65


@dataclass
class VaccinatedModel:
    base: DefenseModel
    rule: DecisionTreeClassifier
    threshold: float
    version: str

    def patch(self, base_scores: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        matched = (
            (self.rule.predict_proba(frame.loc[:, FEATURE_COLUMNS])[:, 1] >= self.threshold)
            & (base_scores < STOP_THRESHOLD)
        )
        return np.where(matched, np.maximum(base_scores, PATCH_SCORE), base_scores)


def _json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_vaccination(
    events: Iterable[dict[str, Any]],
    results_dir: str | Path = ROOT / "results",
    seed: int = 20260827,
    events_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(events)
    output = Path(results_dir)
    model_path = output / "model.pkl"
    survivor_path = output / "arena_survivors.jsonl"
    arena_path = output / "arena_log.json"
    arena_release_path = output / "arena_release.json"
    if not all(
        path.is_file()
        for path in (model_path, survivor_path, arena_path, arena_release_path)
    ):
        raise FileNotFoundError("run training and the five-round arena before vaccination")
    live_hash = model_hash(model_path)
    arena = json.loads(arena_path.read_text(encoding="utf-8"))
    arena_release = json.loads(arena_release_path.read_text(encoding="utf-8"))
    arena_hashes = arena_release.get("artifact_sha256") or {}
    plan = make_split_plan(rows)
    dataset_sha256 = canonical_events_hash(rows)
    split_plan_sha256 = plan.digest()
    if (
        arena_release.get("status") != "passed"
        or arena_release.get("seed") != seed
        or arena.get("seed") != seed
        or arena_release.get("final_model_hash") != live_hash
        or arena_hashes.get("model.pkl") != live_hash
        or arena_hashes.get("arena_log.json") != model_hash(arena_path)
        or arena_hashes.get("arena_survivors.jsonl") != model_hash(survivor_path)
        or arena.get("dataset_sha256") != dataset_sha256
        or arena_release.get("dataset_sha256") != dataset_sha256
        or arena.get("split_plan_sha256") != split_plan_sha256
        or arena_release.get("split_plan_sha256") != split_plan_sha256
        or (arena.get("hardening_gate") or {}).get("passed") is not True
    ):
        raise RuntimeError("vaccination requires the exact passed arena inputs")
    model: DefenseModel = joblib.load(model_path)
    survivors = sorted(_load_jsonl(survivor_path), key=lambda event: str(event["event_id"]))
    if len(survivors) < 2:
        raise ValueError("vaccination requires at least two reserved final-round survivors")

    holdout_count = max(1, len(survivors) // 3)
    vaccine_train = survivors[:-holdout_count]
    survivor_holdout = survivors[-holdout_count:]
    embedder = CachedMiniLM(output / "embeddings.npz")
    memory = PrecedentMemory(embedder, output / "precedent.index", output / "precedent_history.json")
    memory.build(model.history)
    extractor = FeatureExtractor(
        embedder,
        memory,
        model.history,
        trusted_registry=trusted_registry_from_provenance(events_provenance, rows),
    )

    vaccine_batch = extractor.transform(vaccine_train)
    benign_train = [event for event in plan.training if target(event) == 0][: max(len(vaccine_train) * 3, 30)]
    benign_batch = extractor.transform(benign_train)
    training_frame = pd.concat([vaccine_batch.frame, benign_batch.frame], ignore_index=True)
    training_targets = np.r_[np.ones(len(vaccine_train), dtype=int), np.zeros(len(benign_train), dtype=int)]
    rule = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=max(1, min(5, len(vaccine_train))),
        class_weight="balanced",
        random_state=seed,
    ).fit(training_frame.loc[:, FEATURE_COLUMNS], training_targets)
    survivor_probability = rule.predict_proba(vaccine_batch.frame.loc[:, FEATURE_COLUMNS])[:, 1]
    benign_probability = rule.predict_proba(benign_batch.frame.loc[:, FEATURE_COLUMNS])[:, 1]
    threshold = float(np.clip((np.median(survivor_probability) + np.median(benign_probability)) / 2, 0.5, 0.95))
    candidate = VaccinatedModel(model, rule, threshold, "intent-integrity-v2-candidate")

    holdout_batch = extractor.transform(survivor_holdout)
    before_scores = np.asarray(model.predict(holdout_batch)["fused"], dtype=float)
    started = perf_counter()
    after_scores = candidate.patch(before_scores, holdout_batch.frame)
    stopwatch_ms = (perf_counter() - started) * 1_000

    validation_events = list(plan.known + plan.adaptive + plan.benign)
    validation_batch = extractor.transform(validation_events)
    validation_before = np.asarray(model.predict(validation_batch)["fused"], dtype=float)
    validation_after = candidate.patch(validation_before, validation_batch.frame)
    validation_targets = np.asarray([target(event) for event in validation_events], dtype=int)
    old_mask = np.asarray(
        [event["label"]["split"] in {"A", "B"} and bool(target(event)) for event in validation_events],
        dtype=bool,
    )
    e_mask = np.asarray([event["label"]["split"] == "E" for event in validation_events], dtype=bool)
    old_before = float(np.mean(validation_before[old_mask] >= STOP_THRESHOLD)) if np.any(old_mask) else 0.0
    old_after = float(np.mean(validation_after[old_mask] >= STOP_THRESHOLD)) if np.any(old_mask) else 0.0
    benign_before = float(np.mean(validation_before[e_mask] >= STOP_THRESHOLD))
    benign_after = float(np.mean(validation_after[e_mask] >= STOP_THRESHOLD))
    survivor_before = float(np.mean(before_scores >= STOP_THRESHOLD))
    survivor_after = float(np.mean(after_scores >= STOP_THRESHOLD))
    brier_before = float(brier_score_loss(validation_targets, validation_before))
    brier_after = float(brier_score_loss(validation_targets, validation_after))
    gates = {
        "reserved survivor holdout strictly improves": survivor_after > survivor_before and survivor_after == 1.0,
        "old-path regression": old_after >= old_before,
        "benign E stop regression": benign_after <= benign_before,
        "calibration regression": brier_after <= brier_before + 0.02,
    }
    passed = all(gates.values())
    candidate_path = output / "model_v2_candidate.pkl"
    if passed:
        temporary = output / "model_v2_candidate.tmp.pkl"
        joblib.dump(
            {
                "version": candidate.version,
                "base_model_sha256": live_hash,
                "rule": rule,
                "threshold": threshold,
                "patch_score": PATCH_SCORE,
                "status": "awaiting_human_review",
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, candidate_path)
    if model_hash(model_path) != live_hash:
        raise AssertionError("vaccination must never auto-update the live frozen model")

    stages = [
        {"name": "Candidate mini-dataset + shallow rule", "status": "passed"},
        {"name": "Reserved survivor holdout", "status": "passed" if gates["reserved survivor holdout strictly improves"] else "rejected"},
        {"name": "Old-path regression", "status": "passed" if gates["old-path regression"] else "rejected"},
        {"name": "Benign E regression", "status": "passed" if gates["benign E stop regression"] else "rejected"},
        {"name": "Calibration check", "status": "passed" if gates["calibration regression"] else "rejected"},
        {"name": "Human promotion boundary", "status": "awaiting_review" if passed else "rejected"},
    ]
    artifact = {
        "seed": seed,
        **event_provenance(events_provenance),
        "candidate": arena["rounds"][-1]["mutation"],
        "candidate_status": "passed_gates_awaiting_human_review" if passed else "rejected",
        "stages": stages,
        "before_score": round(float(np.mean(before_scores)), 8),
        "after_score": round(float(np.mean(after_scores)), 8),
        "benign_regression_delta": round(benign_after - benign_before, 8),
        "model_version": candidate.version if passed else model.version,
        "candidate_artifact": candidate_path.name if passed else None,
        "promoted_artifact": None,
        "live_model_unchanged": True,
        "live_model_sha256": live_hash,
        "arena_log_sha256": model_hash(arena_path),
        "arena_survivors_sha256": model_hash(survivor_path),
        "arena_release_sha256": model_hash(arena_release_path),
        "stopwatch_ms": round(stopwatch_ms, 6),
        "mini_dataset": {
            "vaccine_training_survivors": len(vaccine_train),
            "reserved_survivor_holdout": len(survivor_holdout),
            "historical_legitimate_A_B": len(benign_train),
            "reserved_benign_E": len(plan.benign),
        },
        "policy_patch": {
            "minimum_score": PATCH_SCORE,
            "rule_threshold": round(threshold, 8),
            "scope": "rule-matched scores below 0.60; HOLD/BLOCK stops payment",
        },
        "before_score_source": "exact frozen post-arena live-model score on never-trained final-round survivors",
        "shallow_tree_rule": export_text(rule, feature_names=list(FEATURE_COLUMNS)),
        "checks": {
            "survivor_stop_rate_before": round(survivor_before, 8),
            "survivor_stop_rate_after": round(survivor_after, 8),
            "old_stop_rate_before": round(old_before, 8),
            "old_stop_rate_after": round(old_after, 8),
            "benign_e_stop_rate_before": round(benign_before, 8),
            "benign_e_stop_rate_after": round(benign_after, 8),
            "brier_before": round(brier_before, 8),
            "brier_after": round(brier_after, 8),
        },
    }
    _json(output / "vaccination.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    rows, provenance = load_events_with_provenance(args.events)
    run_vaccination(rows, args.results, args.seed, events_provenance=provenance)


if __name__ == "__main__":
    main()
