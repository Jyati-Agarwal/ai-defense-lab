"""Measured evaluation and the one-time frozen C/D/E scoring protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve, roc_auc_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from defend.embeddings import CachedMiniLM
from defend.features import FeatureBatch, FeatureExtractor
from defend.harness import (
    FrozenPartitionScorer,
    canonical_events_hash,
    import_firewall_violations,
    make_split_plan,
    model_hash,
    source_tree_hash,
    target,
)
from defend.memory import PrecedentMemory
from defend.models import DefenseModel
from defend.models.bundle import LAYER_ORDER
from defend.policy import policy_decision
from defend.train import (
    EVENT_PROVENANCE_FIELDS,
    ROOT,
    event_provenance,
    load_events_with_provenance,
    trusted_registry_from_provenance,
)
from scripts.release import ReleaseInvalid, _require_evaluation_preregistration


DETECTION_THRESHOLD = 0.30
SHUFFLE_RETRAINS = 50
SHUFFLE_EVAL_PERMUTATIONS = 9_999
SHUFFLE_PROTOCOL_ID = "temporal-a-has-text-conditional-v2-2026-08-27"
EVALUATION_ATTEMPT = "evaluation_attempt_v3.json"
EVALUATION_PREREGISTRATION = "evaluation_v4_preregistration.json"
EVALUATION_RELEASE = "evaluation_v4_release.json"
EVALUATION_SCOPE = {
    "claim": "fresh_row_post_remediation_regression",
    "claim_scope": "post_remediation_regression",
    "fresh_rows_only": True,
    "independent_holdout": False,
    "pristine_heldout_family": False,
    "prior_families_disclosed": True,
    "disclosed_vector_ids": ["GPF-A20", "GPF-C04"],
    "disclosed_family_ids": ["synthetic_identity_bust_out", "ttl_dormancy"],
}
PRIOR_V3_LINEAGE = {
    "prior_preregistration": {
        "path": "evaluation_v3_preregistration.json",
        "sha256": "08b774b653c8edc246443cf8ccefe4d299f07481bd3b23ce1b386befa82f2401",
        "mode": "0444",
    },
    "prior_attempt": {
        "path": "evaluation_attempt_v2.json",
        "sha256": "6e3b2e615f99ff44f99f715ac46725eff550a89cecada0ef5291c7bb5505f5d5",
        "mode": "0444",
    },
    "prior_failure_release": {
        "path": "evaluation_release.json",
        "sha256": "4c312af72f48ae83bb160593df1913ea49c1ab733a6118c4b56f8586b7cb9c69",
        "mode": "0444",
    },
    "prior_model_sha256": "e0add66d9ede329dca413d2ce78318f70717aada26188740498e812e15a79d28",
    "prior_dataset_sha256": "a352a8978407fa888de4af0d1e1c44942ca20397da54eb4a804448439a3a8afd",
    "disclosed_vector_ids": EVALUATION_SCOPE["disclosed_vector_ids"],
    "disclosed_family_ids": EVALUATION_SCOPE["disclosed_family_ids"],
}
V4_REMEDIATION = {
    "utilisation_current_min": 0.90,
    "utilisation_delta_min": 0.70,
    "velocity_1h_min": 5,
    "threshold_changed": False,
    "detection_evaluator_changed": False,
    "evaluation_seed_changed": False,
}
V4_LATENCY_BENCHMARK_SCOPE = "A+B"
V4_PROTECTED_PARTITION_INFERENCE_PASSES = 1
V3_FAILURE_SHA256 = PRIOR_V3_LINEAGE["prior_failure_release"]["sha256"]
ARENA_RELEASE_ARTIFACTS = {
    "model.pkl",
    "arena_survivors.jsonl",
    "arena_log.json",
    "phylogeny.json",
    "asr_curve.png",
}
ARENA_INPUT_EVIDENCE_FIELDS = (
    "dataset_sha256",
    "split_plan_sha256",
    "source_tree_sha256",
    *EVENT_PROVENANCE_FIELDS,
)


@dataclass
class ScoredRow:
    source: dict[str, Any]
    event: dict[str, Any]
    layers: np.ndarray
    fused: float
    latency_ms: float


def _slice(batch: FeatureBatch, position: int) -> FeatureBatch:
    return FeatureBatch(
        batch.frame.iloc[[position]].reset_index(drop=True),
        batch.text_vectors[position : position + 1],
        batch.has_text[position : position + 1],
        [batch.precedents[position]],
    )


def _score_partition(model: DefenseModel, batch: FeatureBatch, events: Iterable[dict[str, Any]]) -> list[ScoredRow]:
    sources = list(events)
    if len(sources) != len(batch.frame):
        raise ValueError("partition events and feature batch must align")
    started = perf_counter()
    prediction = model.predict(batch)
    elapsed_per_event = (perf_counter() - started) * 1_000 / max(1, len(sources))
    predicted_layers = np.asarray(prediction["layers"], dtype=float)
    predicted_fused = np.asarray(prediction["fused"], dtype=float)
    predicted_reasons = list(prediction["reason_codes"])
    output: list[ScoredRow] = []
    for position, source in enumerate(sources):
        layers = predicted_layers[position]
        fused = float(predicted_fused[position])
        reasons = list(predicted_reasons[position])
        event = copy.deepcopy(source)
        event["scores"] = {
            name: round(float(layers[index]), 8) for index, name in enumerate(LAYER_ORDER)
        }
        event["scores"]["fused"] = round(fused, 8)
        event["decision"] = policy_decision(event, fused, reasons, elapsed_per_event)
        if not event["decision"]["reason_codes"]:
            raise AssertionError("every scored decision must have TreeSHAP reason codes")
        output.append(ScoredRow(source, event, layers, fused, elapsed_per_event))
    return output


def _benchmark_singleton_latency(
    model: DefenseModel,
    batch: FeatureBatch,
    sample_count: int = 128,
) -> np.ndarray:
    if len(batch.frame) == 0:
        raise ValueError("latency benchmark requires at least one event")
    positions = np.unique(np.linspace(0, len(batch.frame) - 1, min(sample_count, len(batch.frame)), dtype=int))
    model.predict(_slice(batch, int(positions[0])))
    latencies = []
    for position in positions:
        started = perf_counter()
        model.predict(_slice(batch, int(position)))
        latencies.append((perf_counter() - started) * 1_000)
    return np.asarray(latencies, dtype=float)


def _recall(rows: list[ScoredRow], threshold: float = DETECTION_THRESHOLD) -> float:
    positives = [row for row in rows if target(row.source)]
    if not positives:
        raise ValueError("recall partition contains no positive examples")
    return sum(row.fused >= threshold for row in positives) / len(positives)


def _recall_evidence(rows: list[ScoredRow], threshold: float = DETECTION_THRESHOLD) -> dict[str, Any]:
    positives = [row for row in rows if target(row.source)]
    caught = sum(row.fused >= threshold for row in positives)
    total = len(positives)
    if total == 0:
        raise ValueError("recall partition contains no positive examples")
    proportion = caught / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return {
        "caught": caught,
        "positive_examples": total,
        "recall": round(proportion, 8),
        "ci95_low": round(max(0.0, center - margin), 8),
        "ci95_high": round(min(1.0, center + margin), 8),
    }


def _recall_at_fpr(y: np.ndarray, scores: np.ndarray, maximum_fpr: float) -> float:
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    best = 0.0
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    for threshold in thresholds:
        predicted = scores >= threshold
        fpr = np.sum(predicted & (y == 0)) / negatives
        if fpr <= maximum_fpr + 1e-12:
            best = max(best, float(np.sum(predicted & (y == 1)) / positives))
    return best


def _json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _release_state(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file() and not path.is_symlink():
        try:
            if json.loads(path.read_bytes()).get("status") == "passed":
                return
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
    _json(
        path,
        {
            **payload,
            "schema_version": 4,
            "artifact": "evaluation_v4_release",
            "evaluation_scope": dict(EVALUATION_SCOPE),
        },
    )


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("evaluation receipt write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _require_attempt(
    path: Path,
    evidence: Mapping[str, Any],
    seed: int,
    scoring_counts: Mapping[str, int],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise RuntimeError("evaluation attempt receipt is missing, symbolic, or writable")
    try:
        attempt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("evaluation attempt receipt is invalid") from error
    expected_keys = {
        "schema_version",
        "artifact",
        "status",
        "claim_scope",
        "pristine_heldout_family",
        "independent_holdout",
        "fresh_rows_only",
        "prior_families_disclosed",
        "disclosed_vector_ids",
        "disclosed_family_ids",
        "seed",
        "partitions_to_score_once",
        "scoring_counts",
        *evidence,
    }
    if (
        not isinstance(attempt, dict)
        or set(attempt) != expected_keys
        or attempt.get("schema_version") != 3
        or attempt.get("artifact") != "evaluation_attempt_v3"
        or attempt.get("status") != "consumed_one_shot"
        or attempt.get("claim_scope") != EVALUATION_SCOPE["claim_scope"]
        or attempt.get("pristine_heldout_family") is not False
        or attempt.get("independent_holdout") is not False
        or attempt.get("fresh_rows_only") is not True
        or attempt.get("prior_families_disclosed") is not True
        or attempt.get("disclosed_vector_ids") != EVALUATION_SCOPE["disclosed_vector_ids"]
        or attempt.get("disclosed_family_ids") != EVALUATION_SCOPE["disclosed_family_ids"]
        or type(attempt.get("seed")) is not int
        or attempt.get("seed") != seed
        or attempt.get("partitions_to_score_once") != ["C", "D", "E"]
        or attempt.get("scoring_counts") != dict(scoring_counts)
        or any(attempt.get(key) != value for key, value in evidence.items())
    ):
        raise RuntimeError("evaluation attempt receipt does not match this frozen run")
    return attempt


def _require_preregistration(
    path: Path, expected_sha256: Any, expected_seed: int
) -> dict[str, Any]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o222
    ):
        raise RuntimeError("evaluation V4 requires its immutable preregistration")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("evaluation V4 preregistration is invalid") from error
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if artifact_sha256 != expected_sha256 or not isinstance(payload, dict):
        raise RuntimeError("evaluation V4 preregistration does not match this run")
    lineage = payload.get("lineage")
    remediation = payload.get("remediation")
    generator = payload.get("generator")
    evaluation = payload.get("evaluation")
    custody = payload.get("custody")
    bindings = payload.get("bindings")
    if (
        payload.get("schema_version") != 4
        or payload.get("artifact") != "evaluation_v4_preregistration"
        or payload.get("status") != "predeclared_before_post_remediation_regression"
        or payload.get("claim_scope") != EVALUATION_SCOPE["claim_scope"]
        or payload.get("pristine_heldout_family") is not False
        or payload.get("independent_holdout") is not False
        or payload.get("fresh_rows_only") is not True
        or lineage != PRIOR_V3_LINEAGE
        or remediation != V4_REMEDIATION
        or not isinstance(generator, dict)
        or generator.get("seed") != 20260829
        or generator.get("events") != 50_000
        or generator.get("fraud_rate") != 0.0055
        or generator.get("benign_adversarial_rate") != 0.02
        or generator.get("family_definitions_reused") is not True
        or generator.get("fresh_rows_only") is not True
        or not isinstance(evaluation, dict)
        or evaluation.get("latency_benchmark_scope")
        != V4_LATENCY_BENCHMARK_SCOPE
        or evaluation.get("protected_partition_inference_passes")
        != V4_PROTECTED_PARTITION_INFERENCE_PASSES
        or not isinstance(custody, dict)
        or custody.get("prior_families_disclosed") is not True
        or custody.get("new_rows_not_new_families") is not True
        or custody.get("training_input_physically_mixed") is not True
        or custody.get("v3_evaluation_attempt_created") is not False
        or custody.get("attempt_receipt") != EVALUATION_ATTEMPT
        or custody.get("retry_after_consumption") is not False
        or custody.get("independent_holdout") is not False
        or not isinstance(bindings, dict)
        or bindings.get("parent_failure_sha256") != V3_FAILURE_SHA256
    ):
        raise RuntimeError("evaluation V4 preregistration does not match this run")
    try:
        _require_evaluation_preregistration(
            payload,
            artifact_sha256=artifact_sha256,
            project_root=ROOT,
        )
    except ReleaseInvalid as error:
        raise RuntimeError(
            "evaluation V4 preregistration does not match this run"
        ) from error
    if payload["evaluation"]["seed"] != expected_seed:
        raise RuntimeError("evaluation V4 preregistration does not match this run")
    return payload


def _require_passed_arena_release(
    output: Path, run_evidence: Mapping[str, Any], expected_seed: int
) -> None:
    release_path = output / "arena_release.json"
    if not release_path.is_file():
        raise FileNotFoundError("evaluation requires a valid passed arena release; run `make arena`")
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError("evaluation requires a readable passed arena release") from error
    if release.get("status") != "passed":
        raise RuntimeError("evaluation requires a passed arena release")
    if release.get("seed") != expected_seed:
        raise RuntimeError("arena release seed does not match the frozen evaluation")
    if release.get("final_model_hash") != run_evidence["model_hash"]:
        raise RuntimeError("arena release final model hash does not match model.pkl")
    mismatches = [
        key
        for key in ARENA_INPUT_EVIDENCE_FIELDS
        if release.get(key) != run_evidence[key]
    ]
    if mismatches:
        raise RuntimeError("arena release input evidence does not match: " + ", ".join(mismatches))
    artifact_hashes = release.get("artifact_sha256") or {}
    if not ARENA_RELEASE_ARTIFACTS <= set(artifact_hashes):
        raise RuntimeError("arena release is missing required artifact hashes")
    if any(
        not (output / name).is_file() or model_hash(output / name) != artifact_hashes[name]
        for name in ARENA_RELEASE_ARTIFACTS
    ):
        raise RuntimeError("arena release artifact hashes do not match")


def _plot_pr(path: Path, recall: np.ndarray, precision: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    axis.plot(recall, precision, color="#3bc9ff", linewidth=2.2)
    axis.fill_between(recall, precision, alpha=0.14, color="#3bc9ff")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Frozen-model precision–recall curve")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, transparent=False)
    plt.close(fig)


def _plot_recall(path: Path, recalls: dict[str, float]) -> None:
    names = list(recalls)
    values = np.asarray([recalls[name] for name in names])[None, :]
    fig, axis = plt.subplots(figsize=(max(7.2, len(names) * 0.58), 2.7), dpi=150)
    image = axis.imshow(values, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    axis.set_yticks([0], labels=["Recall"])
    axis.set_xticks(range(len(names)), labels=names, rotation=45, ha="right")
    for position, value in enumerate(values[0]):
        axis.text(position, 0, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _batch_for(extractor: FeatureExtractor, events: tuple[dict[str, Any], ...]) -> FeatureBatch:
    return extractor.transform(events)


def _combine_batches(batches: list[FeatureBatch]) -> FeatureBatch:
    return FeatureBatch(
        pd.concat([batch.frame for batch in batches], ignore_index=True),
        np.vstack([batch.text_vectors for batch in batches]),
        np.concatenate([batch.has_text for batch in batches]),
        [precedent for batch in batches for precedent in batch.precedents],
    )


def _latency_benchmark_batch(
    partition_batches: Mapping[str, FeatureBatch],
) -> FeatureBatch:
    return _combine_batches([partition_batches[name] for name in ("A", "B")])


def _without_deterministic_priors(batch: FeatureBatch) -> FeatureBatch:
    frame = batch.frame.copy()
    frame.loc[:, [
        "structural_floor",
        "layer_semantic",
        "layer_provenance",
        "layer_temporal",
        "layer_economic",
    ]] = 0.0
    return FeatureBatch(frame, batch.text_vectors, batch.has_text, batch.precedents)


def _average_precision_fixed_ranking(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    positives = int(np.sum(y))
    if len(y) != len(values) or positives == 0:
        raise ValueError("average precision needs aligned scores and at least one positive")
    order = np.argsort(values, kind="mergesort")[::-1]
    ordered_scores = values[order]
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(order) - 1]
    return _average_precision_for_order(y, order, group_ends)


def _average_precision_for_order(labels: np.ndarray, order: np.ndarray, group_ends: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    positives = int(np.sum(y))
    if len(y) != len(order) or positives == 0:
        raise ValueError("average precision order needs aligned labels and at least one positive")
    true_positives = np.cumsum(y[order])[group_ends]
    precision = true_positives / (group_ends + 1)
    positive_increments = np.diff(np.r_[0, true_positives])
    return float(np.sum(positive_increments * precision) / positives)


def _score_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    group_ends = np.r_[np.flatnonzero(np.diff(ordered)), len(order) - 1]
    ranks = np.zeros(len(values), dtype=float)
    start = 0
    for end in group_ends:
        ranks[order[start : end + 1]] = (start + end + 2) / 2
        start = int(end) + 1
    return ranks


def _roc_auc_fixed_ranks(labels: np.ndarray, ranks: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    positives = int(np.sum(y))
    negatives = len(y) - positives
    if positives == 0 or negatives == 0 or len(y) != len(ranks):
        raise ValueError("ROC-AUC needs aligned scores and both classes")
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def _conditional_has_text_mask(batch: FeatureBatch, targets: np.ndarray) -> np.ndarray:
    labels = np.asarray(targets, dtype=int)
    mask = np.asarray(batch.has_text, dtype=bool)
    if len(mask) != len(labels):
        raise ValueError("conditional shuffle scope requires aligned modality flags and labels")
    all_positives = int(np.sum(labels))
    retained_positives = int(np.sum(labels[mask]))
    if retained_positives != all_positives:
        raise AssertionError(
            "has-text conditional shuffle scope must retain every temporal-A positive "
            f"({retained_positives}/{all_positives})"
        )
    if len(np.unique(labels[mask])) != 2:
        raise ValueError("has-text conditional shuffle scope requires both classes")
    return mask


def _ensemble_permutation_diagnostics(
    labels: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    ensemble_ap = _average_precision_fixed_ranking(y, values)
    ensemble_ranks = _score_ranks(values)
    ensemble_auc = _roc_auc_fixed_ranks(y, ensemble_ranks)
    ensemble_order = np.argsort(values, kind="mergesort")[::-1]
    ordered_ensemble = values[ensemble_order]
    ensemble_group_ends = np.r_[np.flatnonzero(np.diff(ordered_ensemble)), len(ensemble_order) - 1]
    null_ap = np.empty(SHUFFLE_EVAL_PERMUTATIONS, dtype=float)
    null_auc = np.empty(SHUFFLE_EVAL_PERMUTATIONS, dtype=float)
    for position in range(SHUFFLE_EVAL_PERMUTATIONS):
        permuted = rng.permutation(y)
        null_ap[position] = _average_precision_for_order(permuted, ensemble_order, ensemble_group_ends)
        null_auc[position] = _roc_auc_fixed_ranks(permuted, ensemble_ranks)
    return {
        "pr_auc": ensemble_ap,
        "roc_auc": ensemble_auc,
        "pr_auc_permutation_expectation": float(np.mean(null_ap)),
        "pr_auc_null_median": float(np.median(null_ap)),
        "pr_auc_null_ci99": [float(value) for value in np.quantile(null_ap, [0.005, 0.995])],
        "pr_auc_null_upper_99": float(np.quantile(null_ap, 0.99)),
        "pr_auc_upper_tail_p": float(
            (1 + np.sum(null_ap >= ensemble_ap)) / (SHUFFLE_EVAL_PERMUTATIONS + 1)
        ),
        "roc_auc_null_ci99": [float(value) for value in np.quantile(null_auc, [0.005, 0.995])],
        "roc_auc_two_sided_permutation_p_diagnostic_only": float(
            (1 + np.sum(np.abs(null_auc - 0.5) >= abs(ensemble_auc - 0.5)))
            / (SHUFFLE_EVAL_PERMUTATIONS + 1)
        ),
        "eval_label_permutations": SHUFFLE_EVAL_PERMUTATIONS,
    }


def _shuffled_label_control(
    training_batch: FeatureBatch,
    training_targets: np.ndarray,
    temporal_a_batch: FeatureBatch,
    temporal_a_targets: np.ndarray,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    control_training = _without_deterministic_priors(training_batch)
    control_a = _without_deterministic_priors(temporal_a_batch)
    conditional_mask = _conditional_has_text_mask(control_a, temporal_a_targets)
    conditional_targets = temporal_a_targets[conditional_mask]
    records = []
    score_vectors = []
    for offset in range(SHUFFLE_RETRAINS):
        permutation_seed = seed + offset
        model_seed = seed + 100 + offset
        shuffled = np.random.default_rng(permutation_seed).permutation(training_targets)
        shuffled_model = DefenseModel(seed=model_seed).fit(control_training, shuffled)
        layers = shuffled_model.layer_scores(control_a)
        if shuffled_model.fusion is None:
            raise RuntimeError("shuffled-label fusion is not fitted")
        scores = shuffled_model.fusion.predict_proba(layers)
        record = {
            "offset": offset,
            "permutation_seed": permutation_seed,
            "model_seed": model_seed,
            "pr_auc": _average_precision_fixed_ranking(temporal_a_targets, scores),
            "roc_auc": float(roc_auc_score(temporal_a_targets, scores)),
            "conditional_pr_auc": _average_precision_fixed_ranking(
                conditional_targets, scores[conditional_mask]
            ),
            "conditional_roc_auc": float(
                roc_auc_score(conditional_targets, scores[conditional_mask])
            ),
        }
        records.append(record)
        score_vectors.append(scores)
        if progress is not None:
            progress(record)

    ensemble_scores = np.mean(np.vstack(score_vectors), axis=0)
    individual_ap = np.asarray([record["pr_auc"] for record in records], dtype=float)
    individual_auc = np.asarray([record["roc_auc"] for record in records], dtype=float)
    conditional_individual_ap = np.asarray(
        [record["conditional_pr_auc"] for record in records], dtype=float
    )
    conditional_individual_auc = np.asarray(
        [record["conditional_roc_auc"] for record in records], dtype=float
    )
    v1_rng = np.random.default_rng(seed + 5_000)
    bootstrap_auc_means = np.mean(
        v1_rng.choice(
            individual_auc,
            size=(SHUFFLE_EVAL_PERMUTATIONS, len(individual_auc)),
            replace=True,
        ),
        axis=1,
    )
    v1_diagnostics = _ensemble_permutation_diagnostics(
        temporal_a_targets,
        ensemble_scores,
        v1_rng,
    )
    v2_diagnostics = _ensemble_permutation_diagnostics(
        conditional_targets,
        ensemble_scores[conditional_mask],
        np.random.default_rng(seed + 6_000),
    )
    v1_passed = v1_diagnostics["pr_auc_upper_tail_p"] >= 0.01
    passed = v2_diagnostics["pr_auc_upper_tail_p"] >= 0.01
    positive_mask = temporal_a_targets == 1
    negative_mask = ~positive_mask
    positive_text_count = int(np.sum(conditional_mask & positive_mask))
    negative_text_count = int(np.sum(conditional_mask & negative_mask))
    modality_scores = conditional_mask.astype(float)
    modality_audit = {
        "temporal_a_events": len(temporal_a_targets),
        "temporal_a_positives": int(np.sum(positive_mask)),
        "temporal_a_negatives": int(np.sum(negative_mask)),
        "has_text_events": int(np.sum(conditional_mask)),
        "has_text_positives": positive_text_count,
        "has_text_negatives": negative_text_count,
        "no_text_events": int(np.sum(~conditional_mask)),
        "no_text_positives": int(np.sum((~conditional_mask) & positive_mask)),
        "positive_has_text_rate": float(positive_text_count / max(1, np.sum(positive_mask))),
        "negative_has_text_rate": float(negative_text_count / max(1, np.sum(negative_mask))),
        "all_positives_retained_in_v2": positive_text_count == int(np.sum(positive_mask)),
        "modality_indicator_only_pr_auc": _average_precision_fixed_ranking(
            temporal_a_targets, modality_scores
        ),
        "modality_indicator_only_roc_auc": _roc_auc_fixed_ranks(
            temporal_a_targets, _score_ranks(modality_scores)
        ),
        "v1_mixed_temporal_a": {
            "pr_auc": v1_diagnostics["pr_auc"],
            "roc_auc": v1_diagnostics["roc_auc"],
            "pr_auc_upper_tail_p": v1_diagnostics["pr_auc_upper_tail_p"],
            "status": "PASS" if v1_passed else "FAIL",
            "certification_use": "preserved diagnostic only; label exchangeability is confounded by modality",
        },
        "v2_has_text_conditional": {
            "pr_auc": v2_diagnostics["pr_auc"],
            "roc_auc": v2_diagnostics["roc_auc"],
            "pr_auc_upper_tail_p": v2_diagnostics["pr_auc_upper_tail_p"],
            "status": "PASS" if passed else "FAIL",
            "certification_use": "prospectively predeclared gate",
        },
    }
    return {
        "protocol_id": SHUFFLE_PROTOCOL_ID,
        "scope": (
            "temporal A has-text stratum only; every temporal-A positive retained; "
            "C/D/E excluded from shuffle inference"
        ),
        "procedure": (
            "50 independently seeded training-label permutations of the exact target-disjoint calibrated learned stack; "
            "deterministic direct priors zeroed; mean-score ensemble tested by 9,999 label permutations within the "
            "predeclared temporal-A has-text stratum"
        ),
        "protocol": {
            "training_label_permutations": SHUFFLE_RETRAINS,
            "permutation_seed_formula": "seed + offset, offsets 0..49",
            "model_seed_formula": "seed + 100 + offset, offsets 0..49",
            "v1_evaluation_permutation_seed": seed + 5_000,
            "v2_evaluation_permutation_seed": seed + 6_000,
            "evaluation_label_permutations": SHUFFLE_EVAL_PERMUTATIONS,
            "gate": "one-sided plus-one temporal-A has-text ensemble AP p >= 0.01",
            "frozen_partitions_opened": [],
        },
        "records": records,
        "training_events": len(training_targets),
        "training_positives": int(np.sum(training_targets)),
        "evaluation_events": len(temporal_a_targets),
        "evaluation_positives": int(np.sum(temporal_a_targets)),
        "base_rate": float(np.mean(temporal_a_targets)),
        "conditional_evaluation_events": int(np.sum(conditional_mask)),
        "conditional_evaluation_positives": int(np.sum(conditional_targets)),
        "conditional_base_rate": float(np.mean(conditional_targets)),
        "individual_pr_auc": {
            "mean": float(np.mean(individual_ap)),
            "median": float(np.median(individual_ap)),
            "min": float(np.min(individual_ap)),
            "max": float(np.max(individual_ap)),
        },
        "individual_roc_auc": {
            "mean": float(np.mean(individual_auc)),
            "median": float(np.median(individual_auc)),
            "min": float(np.min(individual_auc)),
            "max": float(np.max(individual_auc)),
            "bootstrap_mean_ci99": [float(value) for value in np.quantile(bootstrap_auc_means, [0.005, 0.995])],
        },
        "individual_conditional_pr_auc": {
            "mean": float(np.mean(conditional_individual_ap)),
            "median": float(np.median(conditional_individual_ap)),
            "min": float(np.min(conditional_individual_ap)),
            "max": float(np.max(conditional_individual_ap)),
        },
        "individual_conditional_roc_auc": {
            "mean": float(np.mean(conditional_individual_auc)),
            "median": float(np.median(conditional_individual_auc)),
            "min": float(np.min(conditional_individual_auc)),
            "max": float(np.max(conditional_individual_auc)),
        },
        "v1_mixed_temporal_a": {
            "status": "PASS" if v1_passed else "FAIL",
            "certification_use": "preserved diagnostic only",
            "ensemble": v1_diagnostics,
        },
        "v2_has_text_conditional": {"status": "PASS" if passed else "FAIL", "ensemble": v2_diagnostics},
        "modality_shortcut_audit": modality_audit,
        "ensemble": v2_diagnostics,
        "gate": {
            "name": "no above-chance within-modality temporal-A PR signal after shuffled training",
            "protocol_id": SHUFFLE_PROTOCOL_ID,
            "scope": "temporal-A has_text=True, all positives retained",
            "ensemble_pr_auc_upper_tail_p_gte_0_01": passed,
            "passed": passed,
            "note": (
                "Below-chance ranking is collapse for this one-sided leakage control; "
                "v1 mixed-modality inference is preserved but is diagnostic-only because modality violates exchangeability."
            ),
        },
    }


def _ablation_score(model: DefenseModel, layers: np.ndarray, name: str) -> np.ndarray:
    values = layers.copy()
    excluded = name.removeprefix("without_")
    values[:, LAYER_ORDER.index(excluded)] = 0.0
    if model.fusion is None:
        raise RuntimeError("fusion is not fitted")
    predicted = model.fusion.predict_proba(values)
    return np.maximum(predicted, values[:, 0])


def _attack_economics(rows: list[ScoredRow]) -> dict[str, Any]:
    positives = [row for row in rows if target(row.source)]
    costs = np.asarray(
        [2_500 + 500 * len((row.source.get("agent_trace") or {}).get("tool_calls") or []) for row in positives],
        dtype=float,
    )
    amounts = np.asarray([(row.source.get("txn") or {}).get("amount_minor", 0) for row in positives], dtype=float)
    legacy_successes = sum(row.event["decision"]["legacy_rules_action"] == "ALLOW" for row in positives)
    blue_successes = sum(row.event["decision"]["action"] == "ALLOW" for row in positives)

    def stage(successes: int, multiplier: float) -> dict[str, Any]:
        rate = successes / max(1, len(positives))
        cost = float(np.mean(costs) * multiplier) if len(costs) else 0.0
        successful_value = float(np.mean(amounts) * rate) if len(amounts) else 0.0
        return {
            "cost_per_attack": round(cost, 4),
            "success_rate": round(rate, 8),
            "cost_per_success": round(float(np.sum(costs) * multiplier / max(successes, 1)), 4),
            "roi": round(successful_value / max(cost, 1.0) - 1, 6),
            "observed_successes": successes,
            "cost_per_success_censored": successes == 0,
        }

    return {"before": stage(legacy_successes, 1.0), "after": stage(blue_successes, 1.25)}


def _evaluate_events(
    events: Iterable[dict[str, Any]],
    results_dir: str | Path = ROOT / "results",
    seed: int = 20260827,
    events_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(results_dir)
    provenance = event_provenance(events_provenance)
    preregistration = _require_preregistration(
        output / EVALUATION_PREREGISTRATION,
        provenance["evaluation_preregistration_sha256"],
        seed,
    )
    rows = list(events)
    model_path = output / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError("train the A+B model before evaluation")
    frozen_hash = model_hash(model_path)
    plan = make_split_plan(rows)
    run_evidence = {
        "model_hash": frozen_hash,
        "dataset_sha256": canonical_events_hash(rows),
        "split_plan_sha256": plan.digest(),
        "source_tree_sha256": source_tree_hash(ROOT),
        **provenance,
    }
    audit_path = output / "frozen_scoring.json"
    release_path = output / EVALUATION_RELEASE
    attempt_path = output / EVALUATION_ATTEMPT
    try:
        _require_passed_arena_release(output, run_evidence, seed)
    except (FileNotFoundError, RuntimeError) as error:
        _release_state(
            release_path,
            {**run_evidence, "status": "failed", "reason": f"arena prerequisite invalid: {error}"},
        )
        raise
    required = (
        output / "metrics.json",
        output / "generalization.json",
        output / "scored_events.jsonl",
        output / "shuffle_control.json",
    )
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if all(audit.get(key) == value for key, value in run_evidence.items()):
            if not all(path.is_file() for path in required):
                _release_state(release_path, {**run_evidence, "status": "failed", "reason": "incomplete cached bundle"})
                raise RuntimeError("frozen scoring cache is incomplete; train a new model hash before rescoring")
            if not release_path.is_file():
                raise RuntimeError("frozen scoring cache is missing its release binding")
            scoring_counts = {name: len(part) for name, part in {
                "A": plan.known,
                "B": plan.adaptive,
                "C": plan.novel,
                "D": plan.heldout,
                "E": plan.benign,
            }.items()}
            _require_attempt(attempt_path, run_evidence, seed, scoring_counts)
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if (
                release.get("schema_version") != 4
                or release.get("artifact") != "evaluation_v4_release"
                or release.get("status") != "passed"
                or release.get("evaluation_scope") != EVALUATION_SCOPE
                or any(
                    release.get(key) != value for key, value in run_evidence.items()
                )
            ):
                raise RuntimeError("frozen scoring release binding is not valid for this run")
            artifact_hashes = release.get("artifact_sha256") or {}
            if not artifact_hashes or any(
                not (output / name).is_file() or model_hash(output / name) != digest
                for name, digest in artifact_hashes.items()
            ):
                _release_state(release_path, {**run_evidence, "status": "failed", "reason": "artifact hash mismatch"})
                raise RuntimeError("frozen scoring release artifact hashes do not match")
            return {
                "metrics": json.loads(required[0].read_text(encoding="utf-8")),
                "generalization": json.loads(required[1].read_text(encoding="utf-8")),
                "cached": True,
            }
        if audit.get("model_hash") == frozen_hash:
            _release_state(release_path, {**run_evidence, "status": "failed", "reason": "evaluation evidence changed"})
            raise RuntimeError("evaluation evidence changed for this frozen model; retrain before scoring again")

    if attempt_path.exists() or attempt_path.is_symlink():
        _release_state(
            release_path,
            {**run_evidence, "status": "failed", "reason": "one-shot evaluation attempt already consumed"},
        )
        raise RuntimeError("one-shot evaluation attempt already consumed; freeze a new regression evaluation revision")

    _release_state(release_path, {**run_evidence, "status": "running"})

    violations = import_firewall_violations(ROOT / "defend")
    if violations:
        raise AssertionError(f"held-out import firewall violated: {violations}")
    missing = [name for name, part in (("A", plan.known), ("B", plan.adaptive), ("C", plan.novel), ("D", plan.heldout), ("E", plan.benign)) if not part]
    if missing:
        raise ValueError(f"evaluation requires populated A/B/C/D/E partitions; missing {missing}")
    model: DefenseModel = joblib.load(model_path)
    embedder = CachedMiniLM(output / "embeddings.npz")
    memory = PrecedentMemory(embedder, output / "precedent.index", output / "precedent_history.json")
    memory.build(model.history)
    extractor = FeatureExtractor(
        embedder,
        memory,
        model.history,
        trusted_registry=trusted_registry_from_provenance(events_provenance, rows),
    )
    training_batch = extractor.transform(plan.training)
    training_targets = np.asarray([target(event) for event in plan.training], dtype=int)
    temporal_a_batch = _batch_for(extractor, plan.known)
    temporal_a_targets = np.asarray([target(event) for event in plan.known], dtype=int)
    shuffle_control = _shuffled_label_control(
        training_batch,
        training_targets,
        temporal_a_batch,
        temporal_a_targets,
        seed,
    )
    if not shuffle_control["gate"]["passed"]:
        ensemble = shuffle_control["ensemble"]
        _json(output / "shuffle_control_attempt.json", {**run_evidence, **shuffle_control})
        _release_state(
            release_path,
            {
                **run_evidence,
                "status": "failed",
                "reason": "shuffled-label control retained above-chance within-modality temporal-A PR signal",
            },
        )
        raise AssertionError(
            f"shuffled-label ensemble retained above-chance within-modality temporal-A PR signal: "
            f"AP={ensemble['pr_auc']:.6f}, null upper99={ensemble['pr_auc_null_upper_99']:.6f}, "
            f"upper-tail p={ensemble['pr_auc_upper_tail_p']:.4f}"
        )

    scoring_counts = {
        "A": len(plan.known),
        "B": len(plan.adaptive),
        "C": len(plan.novel),
        "D": len(plan.heldout),
        "E": len(plan.benign),
    }
    attempt = {
        "schema_version": 3,
        "artifact": "evaluation_attempt_v3",
        "status": "consumed_one_shot",
        "claim_scope": EVALUATION_SCOPE["claim_scope"],
        "pristine_heldout_family": EVALUATION_SCOPE["pristine_heldout_family"],
        "independent_holdout": EVALUATION_SCOPE["independent_holdout"],
        "fresh_rows_only": EVALUATION_SCOPE["fresh_rows_only"],
        "prior_families_disclosed": EVALUATION_SCOPE["prior_families_disclosed"],
        "disclosed_vector_ids": EVALUATION_SCOPE["disclosed_vector_ids"],
        "disclosed_family_ids": EVALUATION_SCOPE["disclosed_family_ids"],
        **run_evidence,
        "seed": seed,
        "partitions_to_score_once": ["C", "D", "E"],
        "scoring_counts": scoring_counts,
    }
    _write_once_json(attempt_path, attempt)
    attempt_sha256 = model_hash(attempt_path)
    _release_state(
        release_path,
        {
            **run_evidence,
            "status": "consumed",
            "attempt_receipt": EVALUATION_ATTEMPT,
            "attempt_receipt_sha256": attempt_sha256,
        },
    )

    parts = {
        "A": (plan.known, temporal_a_batch),
        "B": (plan.adaptive, _batch_for(extractor, plan.adaptive)),
        "C": (plan.novel, _batch_for(extractor, plan.novel)),
        "D": (plan.heldout, _batch_for(extractor, plan.heldout)),
        "E": (plan.benign, _batch_for(extractor, plan.benign)),
    }
    latency_batch = _latency_benchmark_batch(
        {name: partition[1] for name, partition in parts.items()}
    )
    scored = {
        "A": _score_partition(model, parts["A"][1], parts["A"][0]),
        "B": _score_partition(model, parts["B"][1], parts["B"][0]),
    }
    frozen = FrozenPartitionScorer()
    for name in "CDE":
        frozen.claim(name)
        scored[name] = _score_partition(model, parts[name][1], parts[name][0])
    if frozen.scored != {"C", "D", "E"}:
        raise AssertionError("C/D/E were not each scored exactly once")
    if model_hash(model_path) != frozen_hash:
        raise AssertionError("frozen model hash changed while scoring C/D/E")

    all_rows = [row for name in "ABCDE" for row in scored[name]]
    y = np.asarray([target(row.source) for row in all_rows], dtype=int)
    scores = np.asarray([row.fused for row in all_rows], dtype=float)
    layer_values = np.vstack([row.layers for row in all_rows])
    latencies = _benchmark_singleton_latency(model, latency_batch)
    latency_p50 = float(np.percentile(latencies, 50))
    latency_p99 = float(np.percentile(latencies, 99))
    if latency_p99 >= 100:
        raise AssertionError(f"warmed feature-ready singleton p99 {latency_p99:.3f}ms exceeds 100ms target")
    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = float(average_precision_score(y, scores))

    generalization_recalls = {
        "known_recall": _recall(scored["A"]),
        "adaptive_recall": _recall(scored["B"]),
        "novel_composition_recall": _recall(scored["C"]),
        "heldout_family_recall": _recall(scored["D"]),
    }
    generalization_evidence = {
        "known": _recall_evidence(scored["A"]),
        "adaptive": _recall_evidence(scored["B"]),
        "novel_composition": _recall_evidence(scored["C"]),
        "heldout_family": _recall_evidence(scored["D"]),
    }
    generalization = {
        **{key: round(value, 8) for key, value in generalization_recalls.items()},
        "headline": "heldout_family_recall",
        "measured_residual_risk": round(1 - generalization_recalls["heldout_family_recall"], 8),
        "expected_order_holds": all(
            left >= right
            for left, right in zip(generalization_recalls.values(), list(generalization_recalls.values())[1:])
        ),
        "temporal_split": "single global chronological cutoff; every training event precedes every evaluation event",
        "temporal_cutoff": plan.temporal_cutoff,
        "model_hash": frozen_hash,
        "evaluation_scope": dict(EVALUATION_SCOPE),
        "firewall_status": "PASS",
        "contamination_checks": {
            "shuffled_label_pr_auc": round(shuffle_control["ensemble"]["pr_auc"], 8),
            "shuffled_label_summary": (
                "mean-score ensemble of 50 full DefenseModel target-disjoint learned-stack retrains; "
                "deterministic priors excluded; 9,999 label permutations within the predeclared "
                "temporal-A has-text stratum"
            ),
            "shuffled_label_pr_aucs": [
                round(record["conditional_pr_auc"], 8) for record in shuffle_control["records"]
            ],
            "v1_mixed_temporal_a_pr_aucs": [
                round(record["pr_auc"], 8) for record in shuffle_control["records"]
            ],
            "base_rate": round(shuffle_control["conditional_base_rate"], 8),
            "finite_sample_ap_expectation": round(
                shuffle_control["ensemble"]["pr_auc_permutation_expectation"], 8
            ),
            "empirical_null_ci99": [
                round(value, 8) for value in shuffle_control["ensemble"]["pr_auc_null_ci99"]
            ],
            "empirical_null_upper_99": round(shuffle_control["ensemble"]["pr_auc_null_upper_99"], 8),
            "empirical_upper_tail_p": round(shuffle_control["ensemble"]["pr_auc_upper_tail_p"], 8),
            "empirical_permutations": shuffle_control["ensemble"]["eval_label_permutations"],
            "control_scope": shuffle_control["scope"],
            "protocol_id": shuffle_control["protocol_id"],
            "v1_mixed_temporal_a": shuffle_control["v1_mixed_temporal_a"],
            "modality_shortcut_audit": shuffle_control["modality_shortcut_audit"],
            "deterministic_priors_excluded": [
                "structural_floor",
                "layer_semantic",
                "layer_provenance",
                "layer_temporal",
                "layer_economic",
            ],
            "shuffled_label_gate": "PASS",
            "temporal_split": "PASS",
            "import_firewall": "PASS",
            "frozen_hash_invariant": "PASS",
        },
        "scoring_counts": {name: len(scored[name]) for name in "ABCDE"},
        "recall_evidence": generalization_evidence,
    }

    per_id: dict[str, float] = {}
    identifiers = sorted(
        {
            str(row.source["label"]["atlas_id"])
            for row in all_rows
            if target(row.source) and row.source["label"].get("atlas_id")
        }
    )
    for identifier in identifiers:
        group = [row for row in all_rows if str(row.source["label"].get("atlas_id")) == identifier]
        per_id[identifier] = _recall(group)
    zeros = [identifier for identifier, value in per_id.items() if value == 0]
    if zeros:
        raise AssertionError(f"undetected per-vector rows require a feature fix: {zeros}")

    predicted = scores >= DETECTION_THRESHOLD
    matrix = confusion_matrix(y, predicted, labels=[0, 1]).tolist()
    calibration_true, calibration_pred = calibration_curve(y, scores, n_bins=min(10, max(2, len(scores) // 20)), strategy="quantile")
    amounts = np.asarray([(row.source.get("txn") or {}).get("amount_minor", 0) for row in all_rows], dtype=float)
    positive_value = float(np.sum(amounts[y == 1]))
    good_value = float(np.sum(amounts[y == 0]))
    caught_value = float(np.sum(amounts[(y == 1) & predicted]))
    blocked_good_value = float(np.sum(amounts[(y == 0) & (scores >= 0.85)]))

    full_heldout_recall = generalization_recalls["heldout_family_recall"]
    ablations = [{
        "layer": "Full defense",
        "pr_auc": round(pr_auc, 8),
        "heldout_recall": round(full_heldout_recall, 8),
        "pr_auc_drop": 0.0,
        "heldout_recall_drop": 0.0,
    }]
    heldout_offset = sum(len(scored[name]) for name in "ABC")
    heldout_count = len(scored["D"])
    heldout_targets = y[heldout_offset : heldout_offset + heldout_count]
    for name in ("without_semantic", "without_provenance"):
        ablated = _ablation_score(model, layer_values, name)
        heldout_scores = ablated[heldout_offset : heldout_offset + heldout_count]
        ablated_pr_auc = float(average_precision_score(y, ablated))
        ablated_heldout_recall = float(np.mean(heldout_scores[heldout_targets == 1] >= DETECTION_THRESHOLD))
        ablations.append({
            "layer": name.replace("_", " ").title(),
            "pr_auc": round(ablated_pr_auc, 8),
            "heldout_recall": round(ablated_heldout_recall, 8),
            "pr_auc_drop": round(pr_auc - ablated_pr_auc, 8),
            "heldout_recall_drop": round(full_heldout_recall - ablated_heldout_recall, 8),
        })
    if any(row["pr_auc_drop"] <= 0 and row["heldout_recall_drop"] <= 0 for row in ablations[1:]):
        raise AssertionError("semantic and provenance ablations must each reduce measured defense performance")

    weakest = [(identifier, value) for identifier, value in sorted(per_id.items(), key=lambda item: item[1]) if value < 1][:3]
    limitations = [
        {
            "title": f"Residual evidence gap: {identifier}",
            "why": f"Measured recall is {value:.1%}; sparse observable evidence remains.",
            "mitigation": "Route uncertain cases to step-up and add only verified post-review evidence.",
        }
        for identifier, value in weakest
    ] or [
        {
            "title": "No observed regression miss is not proof of perfect coverage",
            "why": (
                f"The disclosed-family D-partition result is {generalization_evidence['heldout_family']['caught']}/"
                f"{generalization_evidence['heldout_family']['positive_examples']} with a 95% lower bound of "
                f"{generalization_evidence['heldout_family']['ci95_low']:.1%}."
            ),
            "mitigation": "Add genuinely independent families and keep the confidence interval beside the point estimate.",
        },
        {
            "title": "Clean-context coercion can erase provenance evidence",
            "why": "A harmful but semantically plausible instruction may leave neither hidden text nor a strong topic shift.",
            "mitigation": "Require scoped tools and step-up for high-impact, low-evidence agent actions.",
        },
        {
            "title": "New legitimate categories can lack precedent support",
            "why": "A sparse verified history can increase manual review even when the purchase is legitimate.",
            "mitigation": "Add only reviewed legitimate outcomes to precedent memory; never self-train from alerts.",
        },
    ]
    stride = max(1, len(recall) // 100)
    metrics = {
        "pr_auc": round(pr_auc, 8),
        "recall_at_0_1_fpr": round(_recall_at_fpr(y, scores, 0.001), 8),
        "recall_at_1_fpr": round(_recall_at_fpr(y, scores, 0.01), 8),
        "brier_score": round(float(brier_score_loss(y, scores)), 8),
        "calibration_curve": [
            {"predicted": round(float(predicted_value), 8), "observed": round(float(observed), 8)}
            for predicted_value, observed in zip(calibration_pred, calibration_true, strict=True)
        ],
        "per_attack_id_recall": {key: round(value, 8) for key, value in per_id.items()},
        "money_weighted": {
            "fraud_value_caught": round(caught_value / max(positive_value, 1.0), 8),
            "good_value_blocked": round(blocked_good_value / max(good_value, 1.0), 8),
        },
        "alerts_per_1000": round(float(np.mean(predicted) * 1_000), 6),
        "latency": {
            "p50_ms": round(latency_p50, 6),
            "p99_ms": round(latency_p99, 6),
            "sample_count": len(latencies),
            "scope": "warmed singleton feature-ready A+B scoring including cached TreeSHAP; HTTP measured separately",
            "latency_benchmark_scope": V4_LATENCY_BENCHMARK_SCOPE,
            "protected_partition_inference_passes": V4_PROTECTED_PARTITION_INFERENCE_PASSES,
        },
        "pr_curve": [
            {"recall": round(float(recall[index]), 8), "precision": round(float(precision[index]), 8)}
            for index in range(0, len(recall), stride)
        ],
        "confusion_matrix": matrix,
        "detection_threshold": DETECTION_THRESHOLD,
        "ablation_table": ablations,
        "precedent_false_positive_reduction": {},
        "economics": _attack_economics(all_rows),
        "known_unknowns": limitations,
    }

    benign_rows = scored["E"]
    benign_batch = parts["E"][1]
    raw_flags = benign_batch.frame["intent_cart_semantic_drift"].to_numpy() >= 0.45
    contextual_flags = benign_batch.frame["contextualized_semantic_drift"].to_numpy() >= 0.45
    raw_rate = float(np.mean(raw_flags))
    contextual_rate = float(np.mean(contextual_flags))
    metrics["precedent_false_positive_reduction"] = {
        "plain_drift_false_positive_rate": round(raw_rate, 8),
        "contextualized_false_positive_rate": round(contextual_rate, 8),
        "absolute_reduction": round(raw_rate - contextual_rate, 8),
        "relative_reduction": round((raw_rate - contextual_rate) / max(raw_rate, 1e-12), 8),
        "sample_count": len(benign_rows),
    }

    precedents: dict[str, Any] = {}
    for position, row in enumerate(benign_rows):
        precedent_evidence = benign_batch.precedents[position]
        if not precedent_evidence["neighbors"]:
            continue
        raw_action = "BLOCK" if raw_flags[position] else "ALLOW"
        contextual_action = row.event["decision"]["action"]
        precedents[str(row.source["event_id"])] = {
            "event_id": row.source["event_id"],
            "query": f"{(row.source.get('mandate_chain') or {}).get('intent', {}).get('prompt_playback', '')} → "
            f"{'+'.join(str(item.get('name', '')) for item in ((row.source.get('mandate_chain') or {}).get('cart', {}).get('items') or []))}",
            "precedent_support": precedent_evidence["support"],
            "baseline_action": raw_action,
            "contextualized_action": contextual_action,
            "verdict": "suppressed" if raw_action == "BLOCK" and contextual_action != "BLOCK" else "confirmed",
            "neighbors": precedent_evidence["neighbors"],
        }

    with tempfile.TemporaryDirectory(prefix=".evaluation-staging-", dir=output) as staging_name:
        staging = Path(staging_name)
        _json(staging / "metrics.json", metrics)
        _json(staging / "generalization.json", generalization)
        _json(staging / "precedents.json", precedents)
        _json(staging / "shuffle_control.json", {**run_evidence, **shuffle_control})
        with (staging / "scored_events.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row.event, ensure_ascii=False) + "\n")
        _plot_pr(staging / "pr_curve.png", recall, precision)
        _plot_recall(staging / "attack_recall_heatmap.png", per_id)
        _json(
            staging / "frozen_scoring.json",
            {
                **run_evidence,
                "attempt_receipt": EVALUATION_ATTEMPT,
                "attempt_receipt_sha256": attempt_sha256,
                "partitions_scored_once": ["C", "D", "E"],
                "weights_unchanged": True,
                "scored_at_seed": seed,
                "scoring_counts": {name: len(scored[name]) for name in "ABCDE"},
                "latency_benchmark_scope": V4_LATENCY_BENCHMARK_SCOPE,
                "protected_partition_inference_passes": V4_PROTECTED_PARTITION_INFERENCE_PASSES,
                "temporal_cutoff": plan.temporal_cutoff,
            },
        )
        promoted_artifact_names = (
            "metrics.json",
            "generalization.json",
            "precedents.json",
            "shuffle_control.json",
            "scored_events.jsonl",
            "pr_curve.png",
            "attack_recall_heatmap.png",
            "frozen_scoring.json",
        )
        artifact_hashes = {
            name: model_hash(staging / name) for name in promoted_artifact_names
        }
        artifact_hashes[EVALUATION_ATTEMPT] = attempt_sha256
        release = {
            **run_evidence,
            "schema_version": 4,
            "artifact": "evaluation_v4_release",
            "status": "passed",
            "evaluation_scope": dict(EVALUATION_SCOPE),
            "attempt_receipt": EVALUATION_ATTEMPT,
            "attempt_receipt_sha256": attempt_sha256,
            "artifact_sha256": artifact_hashes,
        }
        _json(staging / EVALUATION_RELEASE, release)
        for name in promoted_artifact_names:
            os.replace(staging / name, output / name)
        os.replace(staging / EVALUATION_RELEASE, release_path)
    return {"metrics": metrics, "generalization": generalization, "cached": False}


def evaluate_events(
    events: Iterable[dict[str, Any]],
    results_dir: str | Path = ROOT / "results",
    seed: int = 20260827,
    events_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(results_dir)
    release_path = output / EVALUATION_RELEASE
    passed_release = None
    if release_path.is_file() and not release_path.is_symlink():
        try:
            candidate = release_path.read_bytes()
            if json.loads(candidate).get("status") == "passed":
                passed_release = candidate
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
    try:
        return _evaluate_events(events, results_dir, seed, events_provenance)
    except BaseException as error:
        if passed_release is not None:
            temporary = release_path.with_suffix(release_path.suffix + ".tmp")
            temporary.write_bytes(passed_release)
            os.replace(temporary, release_path)
            raise
        attempt_path = output / EVALUATION_ATTEMPT
        if attempt_path.is_file() and not attempt_path.is_symlink():
            try:
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                evidence = {
                    key: attempt[key]
                    for key in (
                        "model_hash",
                        "dataset_sha256",
                        "split_plan_sha256",
                        "source_tree_sha256",
                        *EVENT_PROVENANCE_FIELDS,
                    )
                    if key in attempt
                }
                _release_state(
                    output / EVALUATION_RELEASE,
                    {
                        **evidence,
                        "status": "failed",
                        "attempt_receipt": EVALUATION_ATTEMPT,
                        "attempt_receipt_sha256": model_hash(attempt_path),
                        "reason": f"{type(error).__name__}: {error}",
                    },
                )
            except Exception:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data" / "events.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    rows, provenance = load_events_with_provenance(args.events)
    evaluate_events(rows, args.results, args.seed, events_provenance=provenance)


if __name__ == "__main__":
    main()
