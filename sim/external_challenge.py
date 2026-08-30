"""Sealed, non-operational attack challenge derived from public taxonomies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from defend.harness import model_hash
from defend.policy import action_for_score
from sim.attacks import all_attacks
from sim.mandate_twin import (
    SYNTHETIC_KEY_SCOPE,
    key_registry_sha256,
    observed_event_key_registry,
    validate_synthetic_key_registry,
)
from sim.population import PopulationSimulator
from sim.primitives import AttackDefinition, apply_definition


ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = ROOT / "atlas" / "external_sources.yaml"
FIDELITIES = frozenset({"exact", "behavioral_proxy", "unsupported"})
SCORED_FIDELITIES = frozenset({"exact", "behavioral_proxy"})
TAXONOMY_SOURCES = {
    "mitre_atlas": ("mitre_atlas", "techniques"),
    "owasp_agentic": ("owasp_agentic", "risks"),
    "fraud_classifier": ("federal_reserve_fraud_classifier", "classifications"),
}
WATERMARK = "SAFE ABSTRACT EFFECT / NON-OPERATIONAL"
OFFICIAL_AGENTDOJO_EVIDENCE_PATH = ROOT / "results" / "agentdojo_banking_official.json"
OFFICIAL_AGENTDOJO_EVIDENCE_SHA256 = "aa7d1ffbbbaa7d7e8c94be221f488ebae732cd3d18869354115c625cea055ced"
AGENTDOJO_INTERPRETATION = (
    "Translated synthetic AgentDojo-mapped cases are observable-effect proxy detection only; not an AgentDojo "
    "task-success or ASR reproduction. Private memo values, scheduled-payment state, credentials, and operational "
    "prompts are deliberately absent. Their detector results are reported under recall, separately from the "
    "task-evaluator metrics reported by the hash-pinned, structurally verified AgentDojo banking evidence artifact "
    "under official_agentdojo; authenticity is pending signed release."
)
OFFICIAL_AGENTDOJO_INTERPRETATION = (
    "Utility and attacker-goal success are task-evaluator metrics reported by the hash-pinned, structurally verified "
    "AgentDojo banking evidence artifact; authenticity is pending signed release. They are not our translated "
    "synthetic-event detector score reported under recall."
)

PROFILE_DEFINITIONS: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "prompt_injection": (
        ("context_manip", "intent_manip"),
        {"playback_mismatch": True, "shift_topic": "safe abstract payment-topic deviation"},
    ),
    "translation_injection": (
        ("context_manip", "intent_manip"),
        {"playback_mismatch": True, "shift_topic": "safe abstract translated-topic deviation"},
    ),
    "topic_pivot": (
        ("tool_manip",),
        {"tool_topic": "safe abstract cross-domain payment topic", "topic_shift": 0.94},
    ),
    "playback_mismatch": (
        ("intent_manip", "merchant_manip"),
        {"playback_mismatch": True, "merchant_mode": "substitute_category"},
    ),
    "price_inflation": (
        ("merchant_manip",),
        {"merchant_mode": "price_inflate", "price_ratio": 1.75},
    ),
    "cart_substitution": (
        ("merchant_manip",),
        {"merchant_mode": "substitute_category"},
    ),
    "merchant_substitution": (
        ("merchant_manip",),
        {"merchant_mode": "fake_did"},
    ),
    "delegation_hijack": (
        ("tool_manip",),
        {"delegation_hijack": True, "delegation_depth": 7},
    ),
    "card_velocity": (
        ("reconnaissance", "payment_manip"),
        {"payment_mode": "card_testing", "cards_tested": 19, "attempts": 27},
    ),
    "new_payee": ((), {}),
    "cashout_graph": (
        ("cash_out",),
        {"cashout_velocity": 11, "beneficiary_in_degree": 37},
    ),
    "upi_collect": (
        ("payment_manip",),
        {"payment_mode": "upi_collect"},
    ),
    "dormant_velocity": (
        ("payment_manip", "delayed_exec"),
        {"payment_mode": "card_testing", "cards_tested": 16, "attempts": 22, "dormancy_days": 120},
    ),
    "device_graph": (
        ("cash_out",),
        {"cashout_velocity": 8, "beneficiary_in_degree": 24},
    ),
    "fanout": (
        ("merchant_manip",),
        {"merchant_mode": "fanout", "fanout": 4},
    ),
}

ScoreRows = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atlas_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "atlas").glob("*/*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        specs[str(payload["id"])] = payload
    return specs


def load_mapping(path: str | Path = MAPPING_PATH) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("safety", {}).get("raw_payloads_included") is not False:
        raise ValueError("external challenge mapping must be schema v1 and payload-free")
    if set(payload.get("fidelity_definitions") or {}) != FIDELITIES:
        raise ValueError("external challenge mapping must define every fidelity class")
    for source_key, metadata in payload.get("sources", {}).items():
        if source_key == "local_attack_atlas":
            continue
        digest = str(metadata.get("sha256") or "")
        if not str(metadata.get("url") or "").startswith("https://") or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{source_key}: public source needs an HTTPS URL and lowercase SHA-256 pin")
        if not metadata.get("retrieved_at") or not metadata.get("pin"):
            raise ValueError(f"{source_key}: public source provenance is incomplete")
    specs = _atlas_specs()
    if payload["sources"]["local_attack_atlas"]["sha256"] != _sha256(specs):
        raise ValueError("local attack atlas changed; re-audit the external coverage mapping")
    coverage = payload.get("coverage") or {}
    if set(coverage) != set(specs):
        raise ValueError("external coverage must map every local atlas ID exactly once")
    catalogs = {
        field: set(payload["sources"][source_key][catalog_key])
        for field, (source_key, catalog_key) in TAXONOMY_SOURCES.items()
    }
    existing = {attack.atlas_id for attack in all_attacks(include_heldout=True)}
    exact = set()
    for atlas_id, row in coverage.items():
        fidelity = row.get("fidelity")
        profile = row.get("effect_profile")
        if not row.get("outcome_class") or not row.get("limitation"):
            raise ValueError(f"{atlas_id}: outcome class and fidelity limitation are required")
        if fidelity not in FIDELITIES:
            raise ValueError(f"{atlas_id}: invalid fidelity")
        if fidelity == "exact":
            exact.add(atlas_id)
            if profile != "existing_attack" or atlas_id not in existing:
                raise ValueError(f"{atlas_id}: exact coverage needs an existing concrete attack")
        elif fidelity == "unsupported":
            if profile != "unsupported":
                raise ValueError(f"{atlas_id}: unsupported coverage must not have a runnable profile")
        elif profile not in PROFILE_DEFINITIONS:
            raise ValueError(f"{atlas_id}: unknown behavioral profile {profile}")
        for field, catalog in catalogs.items():
            if not set(row.get(field) or ()) <= catalog:
                raise ValueError(f"{atlas_id}: unknown {field} reference")
    if exact != existing:
        raise ValueError("exact coverage must match the repository's concrete attack catalog")

    tasks = payload.get("agentdojo_cases") or []
    expected_tasks = {f"injection_task_{index}" for index in range(9)}
    if {row.get("task_id") for row in tasks} != expected_tasks or len(tasks) != 9:
        raise ValueError("AgentDojo banking coverage must contain injection_task_0 through injection_task_8")
    for row in tasks:
        if row.get("local_atlas_id") not in specs or row.get("fidelity") not in FIDELITIES:
            raise ValueError(f"invalid AgentDojo task mapping: {row.get('task_id')}")
        profile = row.get("effect_profile")
        if row["fidelity"] == "unsupported":
            if profile != "unsupported":
                raise ValueError("unsupported AgentDojo outcomes cannot have a runnable profile")
        elif profile not in PROFILE_DEFINITIONS:
            raise ValueError(f"unknown AgentDojo behavioral profile: {profile}")
    return payload


def _profile_attack(
    atlas_id: str,
    profile: str,
    outcome_class: str,
    channel: str,
    *,
    language: str = "en",
) -> AttackDefinition:
    primitives, base_parameters = PROFILE_DEFINITIONS[profile]
    parameters = dict(base_parameters)
    if "context_manip" in primitives:
        parameters["language"] = language
    return AttackDefinition(
        atlas_id=atlas_id,
        name=f"safe abstract external challenge: {outcome_class}",
        family=f"external_{outcome_class}",
        split="D",
        primitives=primitives,
        intended_signals=(),
        base_channel=channel,
        parameters=parameters,
    )


def _apply_safe_effect(event: dict[str, Any], profile: str, case_id: str) -> None:
    txn = event["txn"]
    suffix = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    if profile == "new_payee":
        txn["beneficiary_id"] = f"ben_safe_external_{suffix}"
        txn["is_new_beneficiary"] = True
    elif profile == "device_graph":
        txn["shared_device_cluster_size"] = 24
        txn["connected_component_size"] = 31


def _source_keys(source_family: str, mapping: dict[str, Any]) -> list[str]:
    keys = ["agentdojo" if source_family == "agentdojo" else "local_attack_atlas"]
    keys.extend(source_key for field, (source_key, _) in TAXONOMY_SOURCES.items() if mapping.get(field))
    return keys


def _build_case(
    simulator: PopulationSimulator,
    specs: dict[str, dict[str, Any]],
    existing: dict[str, AttackDefinition],
    mapping: dict[str, Any],
    *,
    case_id: str,
    source_family: str,
    source_task_id: str | None,
    repetition: int,
) -> dict[str, Any]:
    atlas_id = str(mapping["local_atlas_id"])
    spec = specs[atlas_id]
    fidelity = str(mapping["fidelity"])
    profile = str(mapping["effect_profile"])
    base = simulator.generate_legitimate(force_channel=str(spec["channel"]))
    applied_primitives: list[str] = []
    if fidelity == "exact":
        attack = existing[atlas_id]
        event = apply_definition(base, attack, registry=simulator.registry)
        applied_primitives = list(attack.primitives)
    elif fidelity == "behavioral_proxy":
        language = ("hi", "mr")[repetition % 2] if profile == "translation_injection" else "en"
        attack = _profile_attack(
            atlas_id,
            profile,
            str(mapping["outcome_class"]),
            str(spec["channel"]),
            language=language,
        )
        event = apply_definition(base, attack, registry=simulator.registry)
        applied_primitives = list(attack.primitives)
        _apply_safe_effect(event, profile, case_id)
    else:
        event = copy.deepcopy(base)

    if profile != "upi_collect":
        event["rail"] = str(spec["rail"])
    event["event_id"] = "evt_ext_" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
    event["label"] = {
        "is_fraud": True,
        "atlas_id": atlas_id,
        "attack_primitives": applied_primitives,
        "attack_family": f"external_{mapping['outcome_class']}",
        "arena_round": None,
        "split": "D",
    }
    taxonomy = {field: list(mapping.get(field) or ()) for field in TAXONOMY_SOURCES}
    event["external_challenge"] = {
        "schema_version": 1,
        "case_id": case_id,
        "evaluation_only": True,
        "watermark": WATERMARK,
        "raw_payload_included": False,
        "source_family": source_family,
        "source_task_id": source_task_id,
        "source_keys": _source_keys(source_family, mapping),
        "outcome_class": str(mapping["outcome_class"]),
        "fidelity": fidelity,
        "local_atlas_id": atlas_id,
        "external_taxonomy": taxonomy,
        "declared_primitives": list(spec["primitives"]),
        "applied_primitives": applied_primitives,
        "effect_profile": profile,
        "limitation": str(mapping["limitation"]),
        "repetition": repetition,
    }
    return event


def generate_challenge(
    *,
    seed: int = 20260827,
    mapping_path: str | Path = MAPPING_PATH,
) -> list[dict[str, Any]]:
    manifest = load_mapping(mapping_path)
    specs = _atlas_specs()
    existing = {attack.atlas_id: attack for attack in all_attacks(include_heldout=True)}
    simulator = PopulationSimulator(seed=seed, customers=160, merchants_per_category=2)
    cases: list[dict[str, Any]] = []
    for atlas_id in sorted(manifest["coverage"]):
        row = {**manifest["coverage"][atlas_id], "local_atlas_id": atlas_id}
        for repetition in range(2):
            cases.append(
                _build_case(
                    simulator,
                    specs,
                    existing,
                    row,
                    case_id=f"external_local_{atlas_id}_{repetition}",
                    source_family="local_atlas",
                    source_task_id=None,
                    repetition=repetition,
                )
            )
    for task in manifest["agentdojo_cases"]:
        crosswalk = manifest["coverage"][task["local_atlas_id"]]
        row = {**task, **{field: list(crosswalk.get(field) or ()) for field in TAXONOMY_SOURCES}}
        cases.append(
            _build_case(
                simulator,
                specs,
                existing,
                row,
                case_id=f"external_agentdojo_{row['task_id']}",
                source_family="agentdojo",
                source_task_id=str(row["task_id"]),
                repetition=0,
            )
        )
    if len(cases) < 100:
        raise AssertionError("external challenge must contain at least 100 deterministic cases")
    return cases


def _event_registry_evidence(
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trusted_registry = validate_synthetic_key_registry(
            manifest.get("key_registry"), manifest.get("synthetic_identity_seed")
        )
    except (OSError, json.JSONDecodeError, AttributeError, ValueError) as error:
        raise RuntimeError("release-bound event key registry is invalid") from error
    registry_sha256 = key_registry_sha256(trusted_registry)
    if (
        not trusted_registry
        or manifest.get("key_registry_scope") != SYNTHETIC_KEY_SCOPE
        or manifest.get("key_registry_sha256") != registry_sha256
    ):
        raise RuntimeError("release-bound event key registry is invalid")
    return trusted_registry, {
        "events_manifest_sha256": model_hash(manifest_path),
        "key_registry_sha256": registry_sha256,
    }


def _require_challenge_registry_coverage(
    events: Iterable[dict[str, Any]],
    trusted_registry: Mapping[str, str],
) -> None:
    try:
        observed = observed_event_key_registry(events)
    except ValueError as error:
        raise RuntimeError("external challenge signer registry is invalid") from error
    if not observed or any(
        trusted_registry.get(principal) != fingerprint
        for principal, fingerprint in observed.items()
    ):
        raise RuntimeError(
            "external challenge signers are not covered by the release-bound registry"
        )


def _score_frozen_model(
    events: list[dict[str, Any]],
    results_dir: Path,
    trusted_registry: Mapping[str, str],
) -> list[dict[str, Any]]:
    import joblib

    from defend.embeddings import CachedMiniLM
    from defend.features import FeatureExtractor
    from defend.memory import PrecedentMemory

    model = joblib.load(results_dir / "model.pkl")
    with tempfile.TemporaryDirectory(prefix="ai-defense-external-challenge-") as temporary:
        embedder = CachedMiniLM(Path(temporary) / "embeddings.npz")
        memory = PrecedentMemory(
            embedder,
            results_dir / "precedent.index",
            results_dir / "precedent_history.json",
        )
        batch = FeatureExtractor(
            embedder,
            memory,
            getattr(model, "history", ()),
            trusted_registry=trusted_registry,
        ).transform(events)
        prediction = model.predict(batch)
    return [
        {
            "fused_score": float(score),
            "action": action_for_score(float(score)),
            "reason_codes": list(reasons),
        }
        for score, reasons in zip(prediction["fused"], prediction["reason_codes"], strict=True)
    ]


def _normalize_scores(rows: Iterable[dict[str, Any]], expected: int) -> list[dict[str, Any]]:
    values = list(rows)
    if len(values) != expected:
        raise ValueError(f"scorer returned {len(values)} rows for {expected} cases")
    normalized = []
    for row in values:
        score = float(row["fused_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("scorer fused_score must be finite and between zero and one")
        expected_action = action_for_score(score)
        action = str(row.get("action") or expected_action)
        if action != expected_action:
            raise ValueError(f"scorer action {action} does not match canonical policy action {expected_action}")
        reasons = [str(reason) for reason in row.get("reason_codes") or ()]
        normalized.append({"fused_score": score, "action": action, "reason_codes": reasons})
    return normalized


def _case_summary(event: dict[str, Any], score: dict[str, Any] | None) -> dict[str, Any]:
    metadata = event["external_challenge"]
    return {
        "case_id": metadata["case_id"],
        "case_sha256": _sha256(event),
        "source_family": metadata["source_family"],
        "source_task_id": metadata["source_task_id"],
        "source_keys": metadata["source_keys"],
        "outcome_class": metadata["outcome_class"],
        "fidelity": metadata["fidelity"],
        "effect_profile": metadata["effect_profile"],
        "limitation": metadata["limitation"],
        "external_taxonomy": metadata["external_taxonomy"],
        "atlas_id": metadata["local_atlas_id"],
        "fused_score": None if score is None else round(float(score["fused_score"]), 8),
        "action": "NOT_SCORED" if score is None else score["action"],
        "reason_codes": [] if score is None else score["reason_codes"],
    }


def _group_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["fidelity"] in SCORED_FIDELITIES]
    detected = [row for row in eligible if row["action"] != "ALLOW"]
    stopped = [row for row in eligible if row["action"] in {"HOLD", "BLOCK"}]
    return {
        "cases": len(rows),
        "eligible": len(eligible),
        "detected": len(detected),
        "detection_recall": None if not eligible else round(len(detected) / len(eligible), 8),
        "stopped": len(stopped),
        "stop_recall": None if not eligible else round(len(stopped) / len(eligible), 8),
    }


def _grouped(
    rows: list[dict[str, Any]],
    keys: Iterable[str],
    member: Callable[[dict[str, Any], str], bool],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    metrics: dict[str, Any] = {}
    actions: dict[str, dict[str, int]] = {}
    for key in sorted(set(keys)):
        selected = [row for row in rows if member(row, key)]
        metrics[key] = _group_metric(selected)
        actions[key] = dict(sorted(Counter(row["action"] for row in selected).items()))
    return metrics, actions


def _public_sources(manifest: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "name",
        "url",
        "repository",
        "paper",
        "preprint",
        "release",
        "published",
        "last_modified",
        "retrieved_at",
        "pin",
        "commit",
        "sha256",
        "license",
        "license_sha256",
        "usage_note",
        "audited_scope",
        "landing_page",
        "classifications",
        "path",
    }
    return {
        key: {field: value for field, value in source.items() if field in allowed}
        for key, source in manifest["sources"].items()
    }


def _official_agentdojo_result(path: Path) -> dict[str, Any]:
    from scripts.agentdojo_official import verify_evidence

    verified = verify_evidence(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != verified["artifact_sha256"] or digest != OFFICIAL_AGENTDOJO_EVIDENCE_SHA256:
        raise RuntimeError("official AgentDojo evidence does not match the canonical artifact SHA-256")
    evidence = json.loads(raw)
    configuration = evidence["configuration"]
    return {
        "evidence": {
            "file": path.resolve().relative_to(ROOT).as_posix(),
            "artifact_sha256": digest,
            "canonical_sha256_match": True,
            "structural_consistency_valid": verified["structural_consistency_valid"],
            "authenticity_verified": verified["authenticity_verified"],
            "authenticity_note": verified["authenticity_note"],
            "raw_logs_verified": verified["raw_logs_verified"],
            "injection_utility_validation": verified["injection_utility_validation"],
        },
        "source": evidence["source"],
        "model": {
            "adapter": configuration["model_adapter"],
            "id": configuration["model_id"],
            "base_url": configuration["base_url"],
        },
        "configuration": configuration,
        "configuration_sha256": verified["configuration_sha256"],
        "run": {
            "protocol": evidence["protocol"],
            "run_id": evidence["run_id"],
            "generated_at": evidence["generated_at"],
            "official_harness_executed": evidence["fidelity"]["official_harness_executed"],
            "official_task_success_evaluators_executed": evidence["fidelity"][
                "official_task_success_evaluators_executed"
            ],
            "attacked_task_pairs": verified["attacked_task_pairs"],
        },
        "utility": verified["utility"],
        "attacker_goal": verified["attacker_goal"],
        "interpretation": OFFICIAL_AGENTDOJO_INTERPRETATION,
    }


def run_challenge(
    results_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    mapping_path: str | Path = MAPPING_PATH,
    official_evidence_path: str | Path = OFFICIAL_AGENTDOJO_EVIDENCE_PATH,
    event_manifest_path: str | Path = ROOT / "data" / "events.manifest.json",
    seed: int = 20260827,
    scorer: ScoreRows | None = None,
) -> dict[str, Any]:
    results = Path(results_dir)
    model_path = results / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"frozen model is missing: {model_path}")
    official_agentdojo = _official_agentdojo_result(Path(official_evidence_path))
    trusted_registry, registry_evidence = _event_registry_evidence(
        Path(event_manifest_path)
    )
    before = model_hash(model_path)
    cases = generate_challenge(seed=seed, mapping_path=mapping_path)
    _require_challenge_registry_coverage(cases, trusted_registry)
    supported = [event for event in cases if event["external_challenge"]["fidelity"] in SCORED_FIDELITIES]
    raw_scores = (
        scorer(supported)
        if scorer is not None
        else _score_frozen_model(supported, results, trusted_registry)
    )
    scores = _normalize_scores(raw_scores, len(supported))
    after = model_hash(model_path)
    if after != before:
        raise RuntimeError("frozen model changed during external challenge scoring")

    scored = iter(scores)
    rows = [
        _case_summary(event, next(scored) if event["external_challenge"]["fidelity"] in SCORED_FIDELITIES else None)
        for event in cases
    ]
    fidelity_metrics, fidelity_actions = _grouped(
        rows,
        (row["fidelity"] for row in rows),
        lambda row, key: row["fidelity"] == key,
    )
    source_keys = {key for row in rows for key in row["source_keys"]}
    source_metrics, source_actions = _grouped(
        rows,
        source_keys,
        lambda row, key: key in row["source_keys"],
    )
    atlas_metrics, atlas_actions = _grouped(
        rows,
        (row["atlas_id"] for row in rows),
        lambda row, key: row["atlas_id"] == key,
    )
    manifest = load_mapping(mapping_path)
    vector_fidelities = Counter(row["fidelity"] for row in manifest["coverage"].values())
    vector_sources = {"local_attack_atlas": len(manifest["coverage"])}
    vector_sources.update(
        {
            source_key: sum(bool(row.get(field)) for row in manifest["coverage"].values())
            for field, (source_key, _) in TAXONOMY_SOURCES.items()
        }
    )
    case_hashes = [row["case_sha256"] for row in rows]
    artifact = {
        "schema_version": 1,
        "protocol": "sealed_external_attack_challenge",
        "seed": seed,
        **registry_evidence,
        "sources": _public_sources(manifest),
        "mapping_sha256": _file_sha256(Path(mapping_path)),
        "case_set_sha256": _sha256(case_hashes),
        "model_hash": before,
        "coverage": {
            "atlas_vectors": len(manifest["coverage"]),
            "by_fidelity": dict(sorted(vector_fidelities.items())),
            "by_source": dict(sorted(vector_sources.items())),
            "fidelity_definitions": manifest["fidelity_definitions"],
        },
        "counts": {
            "total_cases": len(rows),
            "scored_cases": len(supported),
            "unsupported_cases": len(rows) - len(supported),
            "local_atlas_ids": len({row["atlas_id"] for row in rows}),
            "agentdojo_outcomes": sum(row["source_family"] == "agentdojo" for row in rows),
            "by_fidelity": dict(sorted(Counter(row["fidelity"] for row in rows).items())),
            "by_source": {
                key: sum(key in row["source_keys"] for row in rows) for key in sorted(source_keys)
            },
            "by_atlas_id": dict(sorted(Counter(row["atlas_id"] for row in rows).items())),
        },
        "recall": {
            "definitions": {
                "detection": (
                    "eligible exact and behavioral_proxy cases with action STEP_UP, HOLD, or BLOCK; "
                    "unsupported cases are excluded"
                ),
                "stop": (
                    "eligible exact and behavioral_proxy cases with action HOLD or BLOCK; "
                    "unsupported cases are excluded"
                ),
            },
            "overall": _group_metric(rows),
            "by_fidelity": fidelity_metrics,
            "by_source": source_metrics,
            "by_atlas_id": atlas_metrics,
        },
        "action_distribution": {
            "overall": dict(sorted(Counter(row["action"] for row in rows).items())),
            "by_fidelity": fidelity_actions,
            "by_source": source_actions,
            "by_atlas_id": atlas_actions,
        },
        "misses": {
            "scored_allow": [
                {
                    key: row[key]
                    for key in ("case_id", "atlas_id", "source_task_id", "outcome_class", "fidelity", "limitation")
                }
                for row in rows
                if row["fidelity"] in SCORED_FIDELITIES and row["action"] == "ALLOW"
            ],
            "not_stopped": [
                {
                    key: row[key]
                    for key in ("case_id", "atlas_id", "source_task_id", "outcome_class", "fidelity", "limitation", "action")
                }
                for row in rows
                if row["fidelity"] in SCORED_FIDELITIES and row["action"] not in {"HOLD", "BLOCK"}
            ],
            "unsupported": [
                {
                    key: row[key]
                    for key in ("case_id", "atlas_id", "source_task_id", "outcome_class", "limitation")
                }
                for row in rows
                if row["fidelity"] == "unsupported"
            ],
        },
        "case_results": rows,
        "agentdojo_interpretation": AGENTDOJO_INTERPRETATION,
        "official_agentdojo": official_agentdojo,
        "non_contamination": {
            "evaluation_only": True,
            "training_partitions_read": [],
            "training_or_model_selection_writes": [],
            "unsupported_excluded_from_recall": True,
            "model_hash_before": before,
            "model_hash_after": after,
            "frozen_model_hash_invariant": before == after,
            "statement": (
                "Cases are generated after model freeze, scored once, and never imported by training "
                "or model selection."
            ),
        },
        "safety": {
            "watermark": WATERMARK,
            "raw_payloads_included": False,
            "attacker_identifiers_included": False,
            "synthetic_pii_only": True,
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        protected = {
            model_path.resolve(),
            (results / "embeddings.npz").resolve(),
            (results / "precedent.index").resolve(),
            (results / "precedent_history.json").resolve(),
            (results / "training_manifest.json").resolve(),
        }
        if destination.resolve() in protected:
            raise ValueError("external challenge output cannot overwrite a model or training artifact")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            json.dump(artifact, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, destination)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-manifest",
        type=Path,
        default=ROOT / "data" / "events.manifest.json",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    artifact = run_challenge(
        args.results,
        output_path=args.output,
        event_manifest_path=args.event_manifest,
        seed=args.seed,
    )
    print(json.dumps({"output": str(args.output), "counts": artifact["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
