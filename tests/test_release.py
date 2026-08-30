from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.release as release_module

from defend.harness import generator_input_tree_hash, source_tree_hash
from app.backend.artifacts import (
    CUSTODY_FIELD,
    DEFAULT_PRIVATE_KEY,
    PRIVATE_KEY_ENV,
    PUBLIC_KEY_ENV,
    CustodyInvalid,
    generate_signing_key,
)
from scripts.release import (
    AGENTDOJO_OFFICIAL_SHA256,
    ARENA_ARTIFACTS,
    ENTITY_FIDELITY_LIMITATION_SHA256,
    EVALUATION_ARTIFACTS,
    DISCLOSED_FAMILY_IDS,
    DISCLOSED_VECTOR_IDS,
    EVALUATION_V1_FAILURE_SHA256,
    EVALUATION_V2_ATTEMPT_SHA256,
    EVALUATION_V2_ARENA_STOP_SHA256,
    EVALUATION_V2_PREREGISTRATION_SHA256,
    EVALUATION_V3_FAILURE_RELEASE_SHA256,
    EVALUATION_V3_PREREGISTRATION_SHA256,
    MANIFEST,
    PREREGISTERED_RELEASE_SCRIPT,
    RELEASE_POLICY_AMENDMENT,
    RUNTIME_ARTIFACTS,
    V4_LATENCY_BENCHMARK_SCOPE,
    V4_PROTECTED_PARTITION_INFERENCE_PASSES,
    ReleaseInvalid,
    _embedding_model_tree,
    _require_event_manifest,
    build_release as _build_release,
    preregister_evaluation,
    sha256_file,
    verify_release as _verify_release,
    write_release_policy_amendment,
)
from sim.mandate_twin import (
    KeyRegistry,
    build_chain,
    expected_synthetic_key_registry,
    key_registry_sha256,
)


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _private_key(root: Path) -> Path:
    return root.parent / f"{root.name}-release-keys" / "release-ed25519.pem"


def _public_key(root: Path) -> Path:
    return root.parent / f"{root.name}-release-keys" / "release-ed25519.pub.pem"


def _synthetic_registry(*principals: str) -> dict[str, str]:
    return expected_synthetic_key_registry(principals)


def build_release(root: Path) -> dict[str, object]:
    return _build_release(root, _private_key(root))


def verify_release(root: Path) -> dict[str, object]:
    return _verify_release(root, _public_key(root))


def _evaluation_preregistration(root: Path) -> dict[str, object]:
    return {
        "schema_version": 4,
        "artifact": "evaluation_v4_preregistration",
        "status": "predeclared_before_post_remediation_regression",
        "claim_scope": "post_remediation_regression",
        "pristine_heldout_family": False,
        "independent_holdout": False,
        "fresh_rows_only": True,
        "lineage": {
            "prior_preregistration": {
                "path": "evaluation_v3_preregistration.json",
                "sha256": EVALUATION_V3_PREREGISTRATION_SHA256,
                "mode": "0444",
            },
            "prior_attempt": {
                "path": "evaluation_attempt_v2.json",
                "sha256": EVALUATION_V2_ATTEMPT_SHA256,
                "mode": "0444",
            },
            "prior_failure_release": {
                "path": "evaluation_release.json",
                "sha256": EVALUATION_V3_FAILURE_RELEASE_SHA256,
                "mode": "0444",
            },
            "prior_model_sha256": "e0add66d9ede329dca413d2ce78318f70717aada26188740498e812e15a79d28",
            "prior_dataset_sha256": "a352a8978407fa888de4af0d1e1c44942ca20397da54eb4a804448439a3a8afd",
            "disclosed_vector_ids": DISCLOSED_VECTOR_IDS,
            "disclosed_family_ids": DISCLOSED_FAMILY_IDS,
        },
        "remediation": {
            "utilisation_current_min": 0.9,
            "utilisation_delta_min": 0.7,
            "velocity_1h_min": 5,
            "threshold_changed": False,
            "detection_evaluator_changed": False,
            "evaluation_seed_changed": False,
        },
        "generator": {
            "seed": 20260829,
            "events": 50_000,
            "fraud_rate": 0.0055,
            "benign_adversarial_rate": 0.02,
            "family_definitions_reused": True,
            "fresh_rows_only": True,
        },
        "arena": {
            "seed": 20260827,
            "replay_challenge_count": 48,
            "attack_success_threshold": 0.60,
            "training_partitions": ["A", "B"],
            "gate": "final == 0 or final < baseline",
            "zero_residual_is_floor_pass": True,
        },
        "evaluation": {
            "seed": 20260827,
            "detection_threshold": 0.30,
            "training_partitions": ["A", "B"],
            "partitions_scored_once": ["C", "D", "E"],
            "shuffle_retrains": 50,
            "shuffle_permutations": 9_999,
            "latency_benchmark_scope": V4_LATENCY_BENCHMARK_SCOPE,
            "protected_partition_inference_passes": (
                V4_PROTECTED_PARTITION_INFERENCE_PASSES
            ),
        },
        "custody": {
            "prior_families_disclosed": True,
            "new_rows_not_new_families": True,
            "training_input_physically_mixed": True,
            "v3_evaluation_attempt_created": False,
            "attempt_receipt": "evaluation_attempt_v3.json",
            "retry_after_consumption": False,
            "independent_holdout": False,
        },
        "bindings": {
            "sim_population_sha256": sha256_file(root / "sim/population.py"),
            "sim_mandate_twin_sha256": sha256_file(root / "sim/mandate_twin.py"),
            "event_schema_sha256": sha256_file(root / "contracts/event.schema.json"),
            "generator_input_tree_sha256": generator_input_tree_hash(root),
            "defense_source_tree_sha256": source_tree_hash(root),
            "scripts_release_sha256": sha256_file(root / "scripts/release.py"),
            "makefile_sha256": sha256_file(root / "Makefile"),
            "requirements_lock_sha256": sha256_file(root / "requirements.lock"),
            "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "embedding_tree_sha256": _embedding_model_tree(root)["tree_sha256"],
            "parent_failure_sha256": EVALUATION_V3_FAILURE_RELEASE_SHA256,
        },
    }


def _release_tree(root: Path) -> Path:
    for directory in (
        "app/backend",
        "app/frontend/dist/assets",
        "contracts",
        "data/llm_cache/content",
        "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/blobs",
        "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs",
        "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/revision",
        "defend",
        "loop",
        "scripts",
        "sim",
        "sim/attacks/heldout",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "app/__init__.py",
        "app/backend/__init__.py",
        "app/backend/artifacts.py",
        "app/backend/main.py",
        "defend/__init__.py",
        "loop/__init__.py",
        "scripts/release.py",
        "scripts/download_retail_reference.py",
        "scripts/download_retail_ii_reference.py",
        "sim/__init__.py",
        "sim/population.py",
        "sim/entity_fidelity.py",
        "sim/population_v4.py",
        "sim/entity_fidelity_v4.py",
        "sim/mandate_twin.py",
        "sim/primitives.py",
        "sim/content_llm.py",
        "sim/attacks/__init__.py",
        "sim/attacks/heldout/__init__.py",
    ):
        (root / name).write_text(f'# measured runtime: {name}\n', encoding="utf-8")
    (root / "app/frontend/dist/index.html").write_text("<!doctype html>\n", encoding="utf-8")
    (root / "app/frontend/dist/assets/index.js").write_text("export {}\n", encoding="utf-8")
    for name in (
        "Makefile",
        ".dockerignore",
        "Dockerfile",
        "app/frontend/package-lock.json",
        "app/frontend/package.json",
        "requirements.lock",
        "requirements.txt",
    ):
        (root / name).write_text(f"measured packaging: {name}\n", encoding="utf-8")
    (root / "contracts" / "event.schema.json").write_text("{}\n", encoding="utf-8")
    (root / "data" / "market_prices.csv").write_text(
        "mcc,median_minor\n5661,500000\n", encoding="utf-8"
    )
    _json(root / "data/llm_cache/content/cache.json", {"review_status": "approved"})
    fixture_registry = KeyRegistry()
    fixture_chain = build_chain(
        fixture_registry,
        user_id="synthetic",
        merchant_id="synthetic",
        agent_id="agent:synthetic",
        category="shoes",
        budget_minor=100,
        mcc="5661",
        prompt_playback="Synthetic fixture",
        signed_at="2026-08-27T00:00:00Z",
        items=[{"sku": "S-1", "name": "Shoe", "price_minor": 100, "qty": 1}],
        payee_did="did:example:synthetic",
        presence_flag="NOT_PRESENT",
        delegation_depth=1,
    )
    (root / "data/events.jsonl").write_text(
        json.dumps({"mandate_chain": fixture_chain}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _json(
        root / "data" / "events.manifest.json",
        {
            "schema_version": 1,
            "events": 50_000,
            "seed": 20260829,
            "fraud_events": 275,
            "fraud_rate": 0.0055,
            "benign_adversarial_events": 1_000,
            "sha256": sha256_file(root / "data/events.jsonl"),
            "behavior_profile_sha256": None,
            "evaluation_preregistration_sha256": "0" * 64,
            "synthetic_pii": True,
            "api_calls": 0,
            **fixture_registry.manifest_fields(),
        },
    )
    model_root = (
        root
        / "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    )
    (model_root / "blobs/weights").write_bytes(b"measured embedding model")
    (model_root / "refs/main").write_text("revision\n", encoding="utf-8")
    (model_root / "snapshots/revision/model.safetensors").symlink_to("../../blobs/weights")
    results = root / "results"
    results.mkdir()
    reviewed_results = Path(__file__).resolve().parents[1] / "results"
    for name in (
        "evaluation_attempt_v1_consumed_failed.json",
        "evaluation_v2_preregistration.json",
        "evaluation_v2_arena_stop_fail.json",
        "evaluation_v3_preregistration.json",
        "evaluation_attempt_v2.json",
        "evaluation_release.json",
    ):
        (results / name).write_bytes((reviewed_results / name).read_bytes())
        (results / name).chmod(0o444)
    _json(results / "evaluation_v4_preregistration.json", _evaluation_preregistration(root))
    (results / "evaluation_v4_preregistration.json").chmod(0o444)
    event_manifest = json.loads((root / "data/events.manifest.json").read_text(encoding="utf-8"))
    event_manifest["evaluation_preregistration_sha256"] = sha256_file(
        results / "evaluation_v4_preregistration.json"
    )
    _json(root / "data/events.manifest.json", event_manifest)
    for name in sorted(ARENA_ARTIFACTS | EVALUATION_ARTIFACTS):
        (results / name).write_bytes(f"measured-{name}".encode())
    model_sha = sha256_file(results / "model.pkl")
    evidence = {
        "dataset_sha256": "a" * 64,
        "split_plan_sha256": "b" * 64,
        "source_tree_sha256": source_tree_hash(root),
        "events_file_sha256": sha256_file(root / "data/events.jsonl"),
        "events_manifest_sha256": sha256_file(root / "data" / "events.manifest.json"),
        "behavior_profile_sha256": None,
        "evaluation_preregistration_sha256": sha256_file(
            results / "evaluation_v4_preregistration.json"
        ),
        "key_registry_sha256": json.loads(
            (root / "data/events.manifest.json").read_text(encoding="utf-8")
        )["key_registry_sha256"],
    }
    scoring_counts = {"A": 10, "B": 2, "C": 3, "D": 4, "E": 5}
    _json(
        results / "evaluation_attempt_v3.json",
        {
            "schema_version": 3,
            "artifact": "evaluation_attempt_v3",
            "status": "consumed_one_shot",
            "claim_scope": "post_remediation_regression",
            "pristine_heldout_family": False,
            "independent_holdout": False,
            "fresh_rows_only": True,
            "prior_families_disclosed": True,
            "disclosed_vector_ids": DISCLOSED_VECTOR_IDS,
            "disclosed_family_ids": DISCLOSED_FAMILY_IDS,
            **evidence,
            "model_hash": model_sha,
            "seed": 20260827,
            "partitions_to_score_once": ["C", "D", "E"],
            "scoring_counts": scoring_counts,
        },
    )
    (results / "evaluation_attempt_v3.json").chmod(0o444)
    evaluation_attempt_sha256 = sha256_file(results / "evaluation_attempt_v3.json")
    base_model_sha = "d" * 64
    _json(
        results / "arena_log.json",
        {
            "seed": 20260827,
            **evidence,
            "training_partitions": ["A", "B"],
            "input_partitions_materialized": ["A", "B", "C", "D", "E"],
            "partitions_used_for_training_or_feature_extraction": ["A", "B"],
            "base_model_hash": base_model_sha,
            "final_model_hash": model_sha,
            "hardening_gate": {
                "baseline": 0.25,
                "final": 0.0,
                "operator": "final == 0 or final < baseline",
                "passed": True,
                "outcome": "strict_improvement",
            },
        },
    )
    _json(
        results / "frozen_scoring.json",
        {
            **evidence,
            "model_hash": model_sha,
            "attempt_receipt": "evaluation_attempt_v3.json",
            "attempt_receipt_sha256": evaluation_attempt_sha256,
            "partitions_scored_once": ["C", "D", "E"],
            "scoring_counts": scoring_counts,
            "scored_at_seed": 20260827,
            "weights_unchanged": True,
            "latency_benchmark_scope": V4_LATENCY_BENCHMARK_SCOPE,
            "protected_partition_inference_passes": (
                V4_PROTECTED_PARTITION_INFERENCE_PASSES
            ),
        },
    )
    _json(
        results / "generalization.json",
        {
            "known_recall": 0.95,
            "adaptive_recall": 0.90,
            "novel_composition_recall": 0.85,
            "heldout_family_recall": 0.80,
            "headline": "heldout_family_recall",
            "expected_order_holds": True,
            "model_hash": model_sha,
            "evaluation_scope": {
                "claim": "fresh_row_post_remediation_regression",
                "claim_scope": "post_remediation_regression",
                "fresh_rows_only": True,
                "independent_holdout": False,
                "pristine_heldout_family": False,
                "prior_families_disclosed": True,
                "disclosed_vector_ids": DISCLOSED_VECTOR_IDS,
                "disclosed_family_ids": DISCLOSED_FAMILY_IDS,
            },
            "contamination_checks": {
                "shuffled_label_gate": "PASS",
                "temporal_split": "PASS",
                "import_firewall": "PASS",
                "frozen_hash_invariant": "PASS",
            },
        },
    )
    _json(
        results / "metrics.json",
        {
            "detection_threshold": 0.30,
            "latency": {
                "p99_ms": 12.5,
                "sample_count": 32,
                "latency_benchmark_scope": V4_LATENCY_BENCHMARK_SCOPE,
                "protected_partition_inference_passes": (
                    V4_PROTECTED_PARTITION_INFERENCE_PASSES
                ),
            }
        },
    )
    _json(
        results / "arena_release.json",
        {
            "status": "passed",
            "seed": 20260827,
            "base_model_hash": base_model_sha,
            "final_model_hash": model_sha,
            **evidence,
            "artifact_sha256": {
                name: sha256_file(results / name) for name in sorted(ARENA_ARTIFACTS)
            },
        },
    )
    _json(
        results / "evaluation_v4_release.json",
        {
            "schema_version": 4,
            "artifact": "evaluation_v4_release",
            "status": "passed",
            "model_hash": model_sha,
            "attempt_receipt": "evaluation_attempt_v3.json",
            "attempt_receipt_sha256": evaluation_attempt_sha256,
            "evaluation_scope": {
                "claim": "fresh_row_post_remediation_regression",
                "claim_scope": "post_remediation_regression",
                "fresh_rows_only": True,
                "independent_holdout": False,
                "pristine_heldout_family": False,
                "prior_families_disclosed": True,
                "disclosed_vector_ids": DISCLOSED_VECTOR_IDS,
                "disclosed_family_ids": DISCLOSED_FAMILY_IDS,
            },
            **evidence,
            "artifact_sha256": {
                name: sha256_file(results / name) for name in sorted(EVALUATION_ARTIFACTS)
            },
        },
    )
    _json(
        results / "shuffle_control.json",
        {
            "protocol_id": "temporal-a-has-text-conditional-v2-2026-08-27",
            "model_hash": model_sha,
            **evidence,
            "records": [{} for _ in range(50)],
            "evaluation_positives": 35,
            "conditional_evaluation_positives": 35,
            "protocol": {"frozen_partitions_opened": []},
            "v1_mixed_temporal_a": {"status": "FAIL"},
            "gate": {"passed": True},
        },
    )
    evaluation = json.loads((results / "evaluation_v4_release.json").read_text(encoding="utf-8"))
    evaluation["artifact_sha256"]["shuffle_control.json"] = sha256_file(
        results / "shuffle_control.json"
    )
    _json(results / "evaluation_v4_release.json", evaluation)
    for name in ("embeddings.npz", "precedent.index", "precedent_history.json", "atlas_matrix.json"):
        (results / name).write_bytes(f"runtime-{name}".encode())
    _json(
        results / "vaccination.json",
        {
            "live_model_unchanged": True,
            "live_model_sha256": model_sha,
            "seed": 20260827,
            "arena_log_sha256": sha256_file(results / "arena_log.json"),
            "arena_survivors_sha256": sha256_file(results / "arena_survivors.jsonl"),
            "arena_release_sha256": sha256_file(results / "arena_release.json"),
            "events_manifest_sha256": evidence["events_manifest_sha256"],
            "key_registry_sha256": evidence["key_registry_sha256"],
        },
    )
    _json(
        results / "training_manifest.json",
        {**evidence, "training_partitions": ["A", "B"], "model_sha256": base_model_sha},
    )
    _json(
        results / "fidelity.json",
        {
            "schema_version": 1,
            "auc_mean": 0.55,
            "threshold_max_auc": 0.60,
            "events_generated": 200_000,
            "gate_size_required": 200_000,
            "threshold_passed": True,
            "gate_passed": True,
            "reference": {"calibration_evaluation_disjoint": True},
        },
    )
    for name in ENTITY_FIDELITY_LIMITATION_SHA256:
        (results / name).write_bytes((reviewed_results / name).read_bytes())
    official_agentdojo = json.loads(
        (reviewed_results / "agentdojo_banking_official.json").read_text(encoding="utf-8")
    )
    (results / "agentdojo_banking_official.json").write_bytes(
        (reviewed_results / "agentdojo_banking_official.json").read_bytes()
    )
    _json(
        results / "shuffle_null_diagnosis_post_crossfit.json",
        {"gate": {"passed": False}, "ensemble": {"pr_auc_upper_tail_p": 0.007}},
    )
    prior_v1_sha = sha256_file(results / "shuffle_null_diagnosis_post_crossfit.json")
    _json(
        results / "shuffle_control_protocol_v2.json",
        {
            "protocol_id": "temporal-a-has-text-conditional-v2-2026-08-27",
            "status": "predeclared_before_execution",
            "prior_v1": {"sha256": prior_v1_sha, "status": "FAIL"},
            "sealed_partitions_opened": [],
        },
    )
    _json(
        results / "external_attack_challenge.json",
        {
            "schema_version": 1,
            "protocol": "sealed_external_attack_challenge",
            "model_hash": model_sha,
            "events_manifest_sha256": evidence["events_manifest_sha256"],
            "key_registry_sha256": evidence["key_registry_sha256"],
            "non_contamination": {
                "evaluation_only": True,
                "frozen_model_hash_invariant": True,
                "model_hash_before": model_sha,
                "model_hash_after": model_sha,
            },
            "safety": {"raw_payloads_included": False, "synthetic_pii_only": True},
            "official_agentdojo": {
                "evidence": {
                    "file": "results/agentdojo_banking_official.json",
                    "artifact_sha256": AGENTDOJO_OFFICIAL_SHA256,
                    "canonical_sha256_match": True,
                    "structural_consistency_valid": True,
                    "authenticity_verified": False,
                    "authenticity_note": (
                        "Require this artifact SHA-256 in the signed release manifest."
                    ),
                    "raw_logs_verified": False,
                    "injection_utility_validation": "aggregate_only",
                },
                "source": official_agentdojo["source"],
                "configuration": official_agentdojo["configuration"],
                "configuration_sha256": official_agentdojo["configuration_sha256"],
                "run": {
                    "protocol": official_agentdojo["protocol"],
                    "run_id": official_agentdojo["run_id"],
                    "generated_at": official_agentdojo["generated_at"],
                    "official_harness_executed": official_agentdojo["fidelity"][
                        "official_harness_executed"
                    ],
                    "official_task_success_evaluators_executed": official_agentdojo[
                        "fidelity"
                    ]["official_task_success_evaluators_executed"],
                    "attacked_task_pairs": official_agentdojo["plan"][
                        "attacked_task_pairs"
                    ],
                },
                "utility": official_agentdojo["utility"],
                "attacker_goal": official_agentdojo["attacker_goal"],
            },
        },
    )
    generate_signing_key(root, _private_key(root), _public_key(root))
    return results


def _amended_release_tree(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, write_amendment: bool = True
) -> Path:
    results = _release_tree(root)
    release_script = root / "scripts/release.py"
    snapshot = results / PREREGISTERED_RELEASE_SCRIPT
    snapshot.write_bytes(release_script.read_bytes())
    snapshot.chmod(0o444)
    preregistered_script_sha256 = sha256_file(snapshot)
    release_script.write_text("# measured amended release policy\n", encoding="utf-8")

    generalization_path = results / "generalization.json"
    generalization = json.loads(generalization_path.read_text(encoding="utf-8"))
    generalization.update(
        {
            "known_recall": 1.0,
            "adaptive_recall": 1.0,
            "novel_composition_recall": 0.85,
            "heldout_family_recall": 1.0,
            "expected_order_holds": False,
        }
    )
    _json(generalization_path, generalization)

    shuffle_path = results / "shuffle_control.json"
    shuffle = json.loads(shuffle_path.read_text(encoding="utf-8"))
    shuffle["v1_mixed_temporal_a"] = {
        "status": "PASS",
        "certification_use": "preserved diagnostic only",
    }
    _json(shuffle_path, shuffle)

    terminal_path = results / "evaluation_v4_release.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["artifact_sha256"]["generalization.json"] = sha256_file(
        generalization_path
    )
    terminal["artifact_sha256"]["shuffle_control.json"] = sha256_file(shuffle_path)
    _json(terminal_path, terminal)
    terminal_path.chmod(0o444)

    expected_evidence = {
        name: sha256_file(results / name)
        for name in release_module.V4_RELEASE_POLICY_EVIDENCE_SHA256
    }
    monkeypatch.setattr(
        release_module,
        "V4_PREREGISTERED_RELEASE_SCRIPT_SHA256",
        preregistered_script_sha256,
    )
    monkeypatch.setattr(
        release_module, "V4_RELEASE_POLICY_EVIDENCE_SHA256", expected_evidence
    )
    if write_amendment:
        write_release_policy_amendment(root)
    return results


def _preregistration_bootstrap_tree(root: Path) -> Path:
    results = _release_tree(root)
    for name in (
        "evaluation_v4_preregistration.json",
        "evaluation_attempt_v3.json",
        "evaluation_v4_release.json",
    ):
        (results / name).unlink()
    return results


def test_evaluation_preregistration_cli_creates_exact_canonical_contract(
    tmp_path: Path,
) -> None:
    results = _preregistration_bootstrap_tree(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "release.py"),
            "--root",
            str(tmp_path),
            "--preregister-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = results / "evaluation_v4_preregistration.json"
    expected = _evaluation_preregistration(tmp_path)
    expected_bytes = (
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output = json.loads(completed.stdout)
    assert path.read_bytes() == expected_bytes
    assert path.stat().st_mode & 0o777 == 0o444
    assert expected["remediation"]["detection_evaluator_changed"] is False
    assert "evaluator_changed" not in expected["remediation"]
    assert expected["evaluation"]["latency_benchmark_scope"] == "A+B"
    assert expected["evaluation"]["protected_partition_inference_passes"] == 1
    assert output == {
        "status": "created",
        "artifact": path.name,
        "sha256": sha256_file(path),
        "mode": "0444",
    }


def test_evaluation_preregistration_refuses_overwrite(tmp_path: Path) -> None:
    results = _preregistration_bootstrap_tree(tmp_path)
    preregister_evaluation(tmp_path)
    path = results / "evaluation_v4_preregistration.json"
    original = path.read_bytes()

    with pytest.raises(ReleaseInvalid, match="overwrite or extend partial"):
        preregister_evaluation(tmp_path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "name", ("evaluation_attempt_v3.json", "evaluation_v4_release.json")
)
def test_evaluation_preregistration_refuses_partial_downstream_state(
    tmp_path: Path, name: str
) -> None:
    results = _preregistration_bootstrap_tree(tmp_path)
    (results / name).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ReleaseInvalid, match="overwrite or extend partial"):
        preregister_evaluation(tmp_path)

    assert not (results / "evaluation_v4_preregistration.json").exists()


def test_evaluation_preregistration_refuses_partial_target(tmp_path: Path) -> None:
    results = _preregistration_bootstrap_tree(tmp_path)
    path = results / "evaluation_v4_preregistration.json"
    path.write_bytes(b"{")

    with pytest.raises(ReleaseInvalid, match="overwrite or extend partial"):
        preregister_evaluation(tmp_path)

    assert path.read_bytes() == b"{"


def test_evaluation_preregistration_validates_history_before_write(
    tmp_path: Path,
) -> None:
    results = _preregistration_bootstrap_tree(tmp_path)
    historical = results / "evaluation_release.json"
    historical.chmod(0o644)
    historical.write_bytes(historical.read_bytes() + b"\n")
    historical.chmod(0o444)

    with pytest.raises(ReleaseInvalid, match="historical evaluation artifact changed"):
        preregister_evaluation(tmp_path)

    assert not (results / "evaluation_v4_preregistration.json").exists()


def test_evaluation_preregistration_requires_read_only_history(tmp_path: Path) -> None:
    results = _preregistration_bootstrap_tree(tmp_path)
    (results / "evaluation_attempt_v2.json").chmod(0o644)

    with pytest.raises(ReleaseInvalid, match="mode 0444"):
        preregister_evaluation(tmp_path)

    assert not (results / "evaluation_v4_preregistration.json").exists()


def test_build_then_verify_binds_every_required_artifact(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    release = build_release(tmp_path)
    manifest_bytes = (results / "release_manifest.json").read_bytes()

    assert release["status"] == "passed"
    assert release["schema_version"] == 6
    assert release[CUSTODY_FIELD]["algorithm"] == "Ed25519"
    assert release[CUSTODY_FIELD]["signature"]
    assert "app/backend/artifacts.py" in release["runtime_code"]["file_sha256"]
    assert "app/backend/main.py" in release["runtime_code"]["file_sha256"]
    assert "app/frontend/dist/index.html" in release["runtime_code"]["file_sha256"]
    assert "data/events.manifest.json" in release["runtime_data"]["file_sha256"]
    assert "data/market_prices.csv" in release["runtime_data"]["file_sha256"]
    assert any(
        name.endswith("/model.safetensors")
        for name in release["embedding_model"]["file_sha256"]
    )
    assert "Dockerfile" in release["dependency_packaging"]["file_sha256"]
    assert "requirements.lock" in release["dependency_packaging"]["file_sha256"]
    limitation = release["known_limitations"]["entity_behavior_fidelity"]
    assert limitation["status"] == "STOP_FAIL"
    assert limitation["release_eligible"] is False
    assert limitation["runtime_behavior_profile_sha256"] is None
    assert limitation["v4"]["artifact_sha256"] == (
        ENTITY_FIDELITY_LIMITATION_SHA256["entity_fidelity_v4_development_failed.json"]
    )
    assert limitation["mrs2"]["artifact_sha256"] == (
        ENTITY_FIDELITY_LIMITATION_SHA256["entity_fidelity_mrs2_development.json"]
    )
    assert release["known_limitations"]["evaluation_scope"] == {
        "claim": "fresh_row_post_remediation_regression",
        "claim_scope": "post_remediation_regression",
        "fresh_rows_only": True,
        "independent_holdout": False,
        "pristine_heldout_family": False,
        "prior_families_disclosed": True,
        "disclosed_vector_ids": DISCLOSED_VECTOR_IDS,
        "disclosed_family_ids": DISCLOSED_FAMILY_IDS,
    }
    assert release["required_artifact_sha256"]["evaluation_v3_preregistration.json"] == EVALUATION_V3_PREREGISTRATION_SHA256
    assert release["required_artifact_sha256"]["evaluation_attempt_v2.json"] == EVALUATION_V2_ATTEMPT_SHA256
    assert release["required_artifact_sha256"]["evaluation_release.json"] == EVALUATION_V3_FAILURE_RELEASE_SHA256
    for name in (
        "evaluation_v3_preregistration.json",
        "evaluation_attempt_v2.json",
        "evaluation_release.json",
    ):
        assert results.joinpath(name).stat().st_mode & 0o777 == 0o444
    assert (
        release["required_artifact_sha256"]["agentdojo_banking_official.json"]
        == AGENTDOJO_OFFICIAL_SHA256
    )
    assert set(release["required_artifact_sha256"]) == (
        ARENA_ARTIFACTS | EVALUATION_ARTIFACTS | RUNTIME_ARTIFACTS
    )
    assert verify_release(tmp_path) == release
    assert (results / "release_manifest.json").read_bytes() == manifest_bytes


def test_amended_release_signs_and_auto_verifies_nonmonotonic_v4_without_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch)
    (tmp_path / "data/events.jsonl").unlink()

    release = _build_release(
        tmp_path,
        _private_key(tmp_path),
        release_policy_amendment=True,
    )

    limitation = release["known_limitations"][
        "evaluation_release_policy_amendment"
    ]
    assert limitation["expected_order_holds"] is False
    assert limitation["independent_holdout"] is False
    assert set((PREREGISTERED_RELEASE_SCRIPT, RELEASE_POLICY_AMENDMENT)) <= set(
        release["required_artifact_sha256"]
    )
    assert (results / RELEASE_POLICY_AMENDMENT).stat().st_mode & 0o777 == 0o444
    for relative in ("Makefile", "sim/primitives.py", "sim/content_llm.py"):
        (tmp_path / relative).unlink()
    for relative in ("sim/attacks", "data/llm_cache"):
        shutil.rmtree(tmp_path / relative)
    assert _verify_release(tmp_path, _public_key(tmp_path)) == release


def test_ordinary_build_fails_before_collect_when_amendment_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _amended_release_tree(tmp_path, monkeypatch)
    (tmp_path / "data/events.jsonl").unlink()
    monkeypatch.setattr(
        release_module,
        "_collect",
        lambda *args, **kwargs: pytest.fail("ordinary build entered collection"),
    )

    with pytest.raises(ReleaseInvalid, match="explicit amendment signing"):
        _build_release(tmp_path, _private_key(tmp_path))


def test_amendment_does_not_exempt_non_script_preregistration_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _amended_release_tree(tmp_path, monkeypatch)
    (tmp_path / "sim/population.py").write_text("# tampered generator\n", encoding="utf-8")

    with pytest.raises(ReleaseInvalid, match="preregistration|amendment"):
        _build_release(
            tmp_path,
            _private_key(tmp_path),
            release_policy_amendment=True,
        )


def test_amendment_preserves_latency_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch, write_amendment=False)
    metrics_path = results / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["latency"]["p99_ms"] = 120.0
    _json(metrics_path, metrics)
    terminal_path = results / "evaluation_v4_release.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["artifact_sha256"]["metrics.json"] = sha256_file(metrics_path)
    terminal_path.chmod(0o644)
    _json(terminal_path, terminal)
    terminal_path.chmod(0o444)
    monkeypatch.setattr(
        release_module,
        "V4_RELEASE_POLICY_EVIDENCE_SHA256",
        {
            name: sha256_file(results / name)
            for name in release_module.V4_RELEASE_POLICY_EVIDENCE_SHA256
        },
    )
    write_release_policy_amendment(tmp_path)

    with pytest.raises(ReleaseInvalid, match="evaluation gates"):
        _build_release(
            tmp_path,
            _private_key(tmp_path),
            release_policy_amendment=True,
        )


def test_amendment_writer_rejects_stale_evidence_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch, write_amendment=False)
    metrics_path = results / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["detection_threshold"] = 0.31
    _json(metrics_path, metrics)

    with pytest.raises(ReleaseInvalid, match="amendment"):
        write_release_policy_amendment(tmp_path)
    assert not (results / RELEASE_POLICY_AMENDMENT).exists()


@pytest.mark.parametrize(
    "name", (PREREGISTERED_RELEASE_SCRIPT, RELEASE_POLICY_AMENDMENT)
)
def test_amended_release_rejects_missing_custody_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch)
    (results / name).unlink()

    with pytest.raises(ReleaseInvalid, match="missing|amendment"):
        _build_release(
            tmp_path,
            _private_key(tmp_path),
            release_policy_amendment=True,
        )


@pytest.mark.parametrize(
    "name",
    (
        PREREGISTERED_RELEASE_SCRIPT,
        RELEASE_POLICY_AMENDMENT,
        "evaluation_v4_release.json",
    ),
)
def test_amended_release_rejects_writable_custody_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch)
    (results / name).chmod(0o644)

    with pytest.raises(ReleaseInvalid, match="amendment"):
        _build_release(
            tmp_path,
            _private_key(tmp_path),
            release_policy_amendment=True,
        )


def test_amended_release_rejects_inconsistent_reported_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch)
    amendment_path = results / RELEASE_POLICY_AMENDMENT
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["expected_order_holds"] = True
    amendment_path.chmod(0o644)
    _json(amendment_path, amendment)
    amendment_path.chmod(0o444)

    with pytest.raises(ReleaseInvalid, match="amendment"):
        _build_release(
            tmp_path,
            _private_key(tmp_path),
            release_policy_amendment=True,
        )


def test_amended_release_rejects_shuffle_diagnostic_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _amended_release_tree(tmp_path, monkeypatch)
    shuffle_path = results / "shuffle_control.json"
    shuffle = json.loads(shuffle_path.read_text(encoding="utf-8"))
    shuffle["v1_mixed_temporal_a"]["certification_use"] = "certification gate"
    _json(shuffle_path, shuffle)

    with pytest.raises(ReleaseInvalid, match="artifact hash is stale|amendment"):
        _build_release(
            tmp_path,
            _private_key(tmp_path),
            release_policy_amendment=True,
        )


@pytest.mark.parametrize(
    "name",
    (
        "evaluation_v3_preregistration.json",
        "evaluation_attempt_v2.json",
        "evaluation_release.json",
    ),
)
def test_release_rejects_rewritten_historical_evaluation_artifact(
    tmp_path: Path, name: str
) -> None:
    results = _release_tree(tmp_path)
    artifact = results / name
    artifact.chmod(0o644)
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    artifact.chmod(0o444)

    with pytest.raises(ReleaseInvalid, match="historical evaluation artifact changed"):
        build_release(tmp_path)


@pytest.mark.parametrize(
    "name",
    (
        "evaluation_v3_preregistration.json",
        "evaluation_attempt_v2.json",
        "evaluation_release.json",
    ),
)
def test_release_rejects_writable_historical_evaluation_artifact(
    tmp_path: Path, name: str
) -> None:
    results = _release_tree(tmp_path)
    (results / name).chmod(0o644)

    with pytest.raises(ReleaseInvalid, match="mode 0444"):
        build_release(tmp_path)


def test_runtime_release_verification_does_not_require_raw_events(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    release = build_release(tmp_path)
    (tmp_path / "data" / "events.jsonl").unlink()

    assert verify_release(tmp_path) == release


def test_runtime_release_verification_does_not_require_build_only_inputs(
    tmp_path: Path,
) -> None:
    _release_tree(tmp_path)
    release = build_release(tmp_path)
    for relative in ("Makefile", "sim/primitives.py", "sim/content_llm.py"):
        (tmp_path / relative).unlink()
    shutil.rmtree(tmp_path / "sim" / "attacks")
    shutil.rmtree(tmp_path / "data" / "llm_cache")

    assert verify_release(tmp_path) == release


def test_build_release_still_requires_build_only_inputs(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    (tmp_path / "sim" / "primitives.py").unlink()

    with pytest.raises(RuntimeError, match="generator inputs are missing"):
        build_release(tmp_path)


def test_runtime_release_rejects_tampered_signed_preregistration(
    tmp_path: Path,
) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    preregistration = results / "evaluation_v4_preregistration.json"
    payload = json.loads(preregistration.read_text(encoding="utf-8"))
    payload["bindings"]["makefile_sha256"] = "f" * 64
    preregistration.chmod(0o644)
    _json(preregistration, payload)
    preregistration.chmod(0o444)

    with pytest.raises(ReleaseInvalid, match="preregistration does not match"):
        verify_release(tmp_path)


def test_key_generation_command_writes_external_0600_private_key(tmp_path: Path) -> None:
    private_key = tmp_path.parent / f"{tmp_path.name}-cli-keys" / "release.pem"
    public_key = private_key.with_name("release.pub.pem")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "release.py"),
            "--root",
            str(tmp_path),
            "--generate-key",
            str(private_key),
            "--public-key",
            str(public_key),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    assert output == {
        "status": "generated",
        "private_key": str(private_key),
        "public_key": str(public_key),
    }
    assert private_key.stat().st_mode & 0o777 == 0o600
    assert public_key.stat().st_mode & 0o777 == 0o644
    assert not DEFAULT_PRIVATE_KEY.resolve(strict=False).is_relative_to(
        Path(__file__).resolve().parents[1]
    )


def test_verify_rejects_tampered_signed_manifest(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    manifest = json.loads((results / MANIFEST).read_text(encoding="utf-8"))
    manifest["status"] = "tampered"
    _json(results / MANIFEST, manifest)

    with pytest.raises(ReleaseInvalid, match="signature verification failed"):
        verify_release(tmp_path)


def test_verify_rejects_tampered_signature(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    manifest = json.loads((results / MANIFEST).read_text(encoding="utf-8"))
    signature = bytearray(base64.b64decode(manifest[CUSTODY_FIELD]["signature"]))
    signature[0] ^= 1
    manifest[CUSTODY_FIELD]["signature"] = base64.b64encode(signature).decode("ascii")
    _json(results / MANIFEST, manifest)

    with pytest.raises(ReleaseInvalid, match="signature verification failed"):
        verify_release(tmp_path)


def test_verify_rejects_wrong_public_key(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    build_release(tmp_path)
    wrong_private = tmp_path.parent / f"{tmp_path.name}-wrong-keys" / "release.pem"
    wrong_public = wrong_private.with_name("release.pub.pem")
    generate_signing_key(tmp_path, wrong_private, wrong_public)

    with pytest.raises(ReleaseInvalid, match="does not match the signed release"):
        _verify_release(tmp_path, wrong_public)


def test_verify_rejects_public_key_inside_repository(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    build_release(tmp_path)
    inside = tmp_path / "release.pub.pem"
    inside.write_bytes(_public_key(tmp_path).read_bytes())
    inside.chmod(0o644)

    with pytest.raises(ReleaseInvalid, match="outside the repository"):
        _verify_release(tmp_path, inside)


def test_environment_configured_keys_build_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release_tree(tmp_path)
    monkeypatch.setenv(PRIVATE_KEY_ENV, str(_private_key(tmp_path)))
    monkeypatch.setenv(PUBLIC_KEY_ENV, str(_public_key(tmp_path)))

    release = _build_release(tmp_path)

    assert _verify_release(tmp_path) == release


def test_missing_private_or_public_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _release_tree(tmp_path)
    monkeypatch.delenv(PRIVATE_KEY_ENV, raising=False)
    with pytest.raises(ReleaseInvalid, match="private key is not configured"):
        _build_release(tmp_path)
    assert not (results / MANIFEST).exists()

    build_release(tmp_path)
    monkeypatch.delenv(PUBLIC_KEY_ENV, raising=False)
    with pytest.raises(ReleaseInvalid, match="public key is not configured"):
        _verify_release(tmp_path)


def test_verify_rejects_missing_signature(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    manifest = json.loads((results / MANIFEST).read_text(encoding="utf-8"))
    del manifest[CUSTODY_FIELD]
    _json(results / MANIFEST, manifest)

    with pytest.raises(ReleaseInvalid, match="no valid signed custody record"):
        verify_release(tmp_path)


def test_key_paths_reject_repository_paths_symlinks_and_unsafe_modes(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "secrets" / "release.pem"
    with pytest.raises(CustodyInvalid, match="outside the repository"):
        generate_signing_key(tmp_path, inside)
    assert not inside.parent.exists()

    with pytest.raises(CustodyInvalid, match="absolute"):
        generate_signing_key(tmp_path, Path("relative-release.pem"))

    _release_tree(tmp_path)
    private_bytes = _private_key(tmp_path).read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_signing_key(tmp_path, _private_key(tmp_path), _public_key(tmp_path))
    assert _private_key(tmp_path).read_bytes() == private_bytes

    private_alias = _private_key(tmp_path).with_name("private-alias.pem")
    public_alias = _public_key(tmp_path).with_name("public-alias.pem")
    private_alias.symlink_to(_private_key(tmp_path))
    public_alias.symlink_to(_public_key(tmp_path))
    victim = _private_key(tmp_path).with_name("victim.pem")
    victim.write_bytes(b"must-not-change")
    generation_alias = _private_key(tmp_path).with_name("generation-alias.pem")
    generation_alias.symlink_to(victim)
    with pytest.raises(CustodyInvalid, match="must not contain symlinks"):
        generate_signing_key(tmp_path, generation_alias)
    assert victim.read_bytes() == b"must-not-change"

    with pytest.raises(ReleaseInvalid, match="must not contain symlinks"):
        _build_release(tmp_path, private_alias)

    private_hardlink = tmp_path / "private-hardlink.pem"
    os.link(_private_key(tmp_path), private_hardlink)
    with pytest.raises(ReleaseInvalid, match="exactly one hard link"):
        _build_release(tmp_path, _private_key(tmp_path))
    private_hardlink.unlink()

    _private_key(tmp_path).parent.chmod(0o777)
    with pytest.raises(ReleaseInvalid, match="directory is not a trusted location"):
        _build_release(tmp_path, _private_key(tmp_path))
    _private_key(tmp_path).parent.chmod(0o700)

    build_release(tmp_path)
    with pytest.raises(ReleaseInvalid, match="must not contain symlinks"):
        _verify_release(tmp_path, public_alias)

    _public_key(tmp_path).chmod(0o666)
    with pytest.raises(ReleaseInvalid, match="must not be group or other writable"):
        _verify_release(tmp_path, _public_key(tmp_path))
    _public_key(tmp_path).chmod(0o644)

    _private_key(tmp_path).chmod(0o644)
    with pytest.raises(ReleaseInvalid, match="permissions"):
        _build_release(tmp_path, _private_key(tmp_path))


def test_verify_rejects_tampered_required_artifact(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    (results / "metrics.json").write_bytes(b"tampered")

    with pytest.raises(ReleaseInvalid, match="hash is stale"):
        verify_release(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        "app/backend/main.py",
        "app/frontend/dist/index.html",
        "sim/population.py",
        "data/events.manifest.json",
        "data/market_prices.csv",
        "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/blobs/weights",
        "Dockerfile",
        "requirements.lock",
    ),
)
def test_verify_rejects_tampered_runtime_or_packaging_file(
    tmp_path: Path, relative_path: str
) -> None:
    _release_tree(tmp_path)
    build_release(tmp_path)
    (tmp_path / relative_path).write_bytes(b"tampered")

    with pytest.raises(ReleaseInvalid, match="stale|provenance bindings|does not match"):
        verify_release(tmp_path)


def test_verify_rejects_tampered_release_bound_key_registry(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    build_release(tmp_path)
    manifest_path = tmp_path / "data/events.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["key_registry"]["user:synthetic"] = "sha256:" + "e" * 64
    _json(manifest_path, manifest)

    with pytest.raises(ReleaseInvalid, match="event manifest SHA-256"):
        verify_release(tmp_path)


def test_verify_rejects_unbound_native_import_shadow(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    build_release(tmp_path)
    (tmp_path / "app/backend/main.cpython-311-x86_64-linux-gnu.so").write_bytes(
        b"unbound native module"
    )

    with pytest.raises(ReleaseInvalid, match="runtime code directory contains an unbound file"):
        verify_release(tmp_path)


def test_missing_artifact_cannot_replace_last_good_manifest(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    previous = (results / "release_manifest.json").read_bytes()
    (results / "embeddings.npz").unlink()

    with pytest.raises(ReleaseInvalid, match="embeddings.npz"):
        build_release(tmp_path)
    assert (results / "release_manifest.json").read_bytes() == previous


def test_stale_upstream_model_evidence_is_rejected(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    evaluation = json.loads((results / "evaluation_v4_release.json").read_text(encoding="utf-8"))
    evaluation["model_hash"] = "0" * 64
    _json(results / "evaluation_v4_release.json", evaluation)

    with pytest.raises(ReleaseInvalid, match="model evidence"):
        verify_release(tmp_path)


def test_upstream_release_cannot_smuggle_an_extra_artifact(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    (results / "unreviewed.bin").write_bytes(b"unreviewed")
    arena = json.loads((results / "arena_release.json").read_text(encoding="utf-8"))
    arena["artifact_sha256"]["unreviewed.bin"] = sha256_file(results / "unreviewed.bin")
    _json(results / "arena_release.json", arena)

    with pytest.raises(ReleaseInvalid, match="unexpected artifact hashes"):
        build_release(tmp_path)


def test_failed_measured_fidelity_gate_cannot_build(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    fidelity = json.loads((results / "fidelity.json").read_text(encoding="utf-8"))
    fidelity["gate_passed"] = False
    _json(results / "fidelity.json", fidelity)

    with pytest.raises(ReleaseInvalid, match="200k gate"):
        build_release(tmp_path)
    assert not (results / "release_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (("threshold_max_auc", 1.0), ("events_generated", 200_000.9)),
)
def test_fidelity_gate_constants_cannot_be_weakened(
    tmp_path: Path, field: str, value: object
) -> None:
    results = _release_tree(tmp_path)
    fidelity = json.loads((results / "fidelity.json").read_text(encoding="utf-8"))
    fidelity[field] = value
    _json(results / "fidelity.json", fidelity)

    with pytest.raises(ReleaseInvalid, match="200k gate"):
        build_release(tmp_path)


def test_tampered_entity_stop_fail_evidence_cannot_build(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    failure = json.loads(
        (results / "entity_fidelity_mrs2_development.json").read_text(encoding="utf-8")
    )
    failure["status"] = "PASS"
    _json(results / "entity_fidelity_mrs2_development.json", failure)

    with pytest.raises(ReleaseInvalid, match="immutable reviewed set"):
        build_release(tmp_path)


def test_agentdojo_external_reference_must_match_signed_evidence(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    challenge = json.loads(
        (results / "external_attack_challenge.json").read_text(encoding="utf-8")
    )
    challenge["official_agentdojo"]["evidence"]["artifact_sha256"] = "0" * 64
    _json(results / "external_attack_challenge.json", challenge)

    with pytest.raises(ReleaseInvalid, match="external attack challenge is stale or unsafe"):
        build_release(tmp_path)


@pytest.mark.parametrize(
    "field", ("events_manifest_sha256", "key_registry_sha256")
)
def test_external_challenge_must_match_event_registry_evidence(
    tmp_path: Path, field: str
) -> None:
    results = _release_tree(tmp_path)
    challenge = json.loads(
        (results / "external_attack_challenge.json").read_text(encoding="utf-8")
    )
    challenge[field] = "0" * 64
    _json(results / "external_attack_challenge.json", challenge)

    with pytest.raises(
        ReleaseInvalid, match="external attack challenge is stale or unsafe"
    ):
        build_release(tmp_path)


def test_verify_rejects_tampered_agentdojo_evidence_bytes(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    build_release(tmp_path)
    official = json.loads(
        (results / "agentdojo_banking_official.json").read_text(encoding="utf-8")
    )
    official["generated_at"] = "tampered"
    _json(results / "agentdojo_banking_official.json", official)

    with pytest.raises(ReleaseInvalid, match="external attack challenge is stale or unsafe"):
        verify_release(tmp_path)


def test_profiled_event_provenance_cannot_hide_failed_entity_fidelity(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    for name in ("arena_release.json", "evaluation_v4_release.json", "arena_log.json"):
        payload = json.loads((results / name).read_text(encoding="utf-8"))
        payload["behavior_profile_sha256"] = "f" * 64
        _json(results / name, payload)

    with pytest.raises(ReleaseInvalid, match="behavior_profile_sha256"):
        build_release(tmp_path)


def test_null_behavior_profile_provenance_must_be_explicit(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    for name in ("arena_release.json", "evaluation_v4_release.json"):
        payload = json.loads((results / name).read_text(encoding="utf-8"))
        del payload["behavior_profile_sha256"]
        _json(results / name, payload)

    with pytest.raises(ReleaseInvalid, match="behavior_profile_sha256"):
        build_release(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("seed", 1), ("fraud_events", 274), ("benign_adversarial_events", 999)),
)
def test_event_manifest_rejects_noncanonical_generation(
    field: str, value: object
) -> None:
    key_registry = _synthetic_registry("user:synthetic")
    payload = {
        "schema_version": 1,
        "events": 50_000,
        "seed": 20260829,
        "fraud_events": 275,
        "fraud_rate": 0.0055,
        "benign_adversarial_events": 1_000,
        "sha256": "c" * 64,
        "behavior_profile_sha256": None,
        "evaluation_preregistration_sha256": "f" * 64,
        "synthetic_pii": True,
        "api_calls": 0,
        "synthetic_identity_seed": "ai-defense-lab:public-synthetic-identities:v1",
        "key_registry_scope": "synthetic fixtures only; public deterministic keys; never production credentials",
        "key_registry": key_registry,
        "key_registry_sha256": key_registry_sha256(key_registry),
    }
    payload[field] = value

    with pytest.raises(ReleaseInvalid, match="key registry|synthetic provenance"):
        _require_event_manifest(
            payload,
            manifest_sha256="e" * 64,
            events_sha256="c" * 64,
            evaluation_preregistration_sha256="f" * 64,
            registry_sha256=key_registry_sha256(key_registry),
            observed_registry=key_registry,
        )


@pytest.mark.parametrize(
    "key_registry",
    (
        None,
        {},
        {"user:synthetic": "not-a-fingerprint"},
        {"unscoped": "sha256:" + "d" * 64},
    ),
)
def test_event_manifest_rejects_missing_or_invalid_key_registry(
    key_registry: object,
) -> None:
    valid_registry = _synthetic_registry("user:synthetic")
    payload = {
        "schema_version": 1,
        "events": 50_000,
        "seed": 20260829,
        "fraud_events": 275,
        "fraud_rate": 0.0055,
        "benign_adversarial_events": 1_000,
        "sha256": "c" * 64,
        "behavior_profile_sha256": None,
        "evaluation_preregistration_sha256": "f" * 64,
        "synthetic_pii": True,
        "api_calls": 0,
        "synthetic_identity_seed": "ai-defense-lab:public-synthetic-identities:v1",
        "key_registry_scope": "synthetic fixtures only; public deterministic keys; never production credentials",
        "key_registry": key_registry,
        "key_registry_sha256": key_registry_sha256(valid_registry),
    }

    with pytest.raises(ReleaseInvalid, match="key registry|synthetic provenance"):
        _require_event_manifest(
            payload,
            manifest_sha256="e" * 64,
            events_sha256="c" * 64,
            evaluation_preregistration_sha256="f" * 64,
            registry_sha256=key_registry_sha256(valid_registry),
            observed_registry=valid_registry,
        )


def test_event_manifest_rejects_partial_observed_registry() -> None:
    declared_registry = _synthetic_registry("user:synthetic")
    observed_registry = _synthetic_registry(
        "user:synthetic", "merchant:synthetic"
    )
    payload = {
        "schema_version": 1,
        "events": 50_000,
        "seed": 20260829,
        "fraud_events": 275,
        "fraud_rate": 0.0055,
        "benign_adversarial_events": 1_000,
        "sha256": "c" * 64,
        "behavior_profile_sha256": None,
        "evaluation_preregistration_sha256": "f" * 64,
        "synthetic_pii": True,
        "api_calls": 0,
        "synthetic_identity_seed": "ai-defense-lab:public-synthetic-identities:v1",
        "key_registry_scope": "synthetic fixtures only; public deterministic keys; never production credentials",
        "key_registry": declared_registry,
        "key_registry_sha256": key_registry_sha256(declared_registry),
    }

    with pytest.raises(ReleaseInvalid, match="synthetic provenance"):
        _require_event_manifest(
            payload,
            manifest_sha256="e" * 64,
            events_sha256="c" * 64,
            evaluation_preregistration_sha256="f" * 64,
            registry_sha256=key_registry_sha256(declared_registry),
            observed_registry=observed_registry,
        )


def test_failed_current_shuffle_gate_cannot_build(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    control = json.loads((results / "shuffle_control.json").read_text(encoding="utf-8"))
    control["gate"]["passed"] = False
    _json(results / "shuffle_control.json", control)
    evaluation = json.loads((results / "evaluation_v4_release.json").read_text(encoding="utf-8"))
    evaluation["artifact_sha256"]["shuffle_control.json"] = sha256_file(
        results / "shuffle_control.json"
    )
    _json(results / "evaluation_v4_release.json", evaluation)

    with pytest.raises(ReleaseInvalid, match="shuffle-control"):
        build_release(tmp_path)


def test_release_rejects_training_arena_dependency_mismatch(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    arena = json.loads((results / "arena_release.json").read_text(encoding="utf-8"))
    arena["base_model_hash"] = "0" * 64
    _json(results / "arena_release.json", arena)

    with pytest.raises(ReleaseInvalid, match="training/arena dependency"):
        build_release(tmp_path)


def test_release_rejects_noncanonical_arena_seed(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    arena_log = json.loads((results / "arena_log.json").read_text(encoding="utf-8"))
    arena_log["seed"] = 1
    _json(results / "arena_log.json", arena_log)
    arena_release = json.loads(
        (results / "arena_release.json").read_text(encoding="utf-8")
    )
    arena_release["seed"] = 1
    arena_release["artifact_sha256"]["arena_log.json"] = sha256_file(
        results / "arena_log.json"
    )
    _json(results / "arena_release.json", arena_release)

    with pytest.raises(ReleaseInvalid, match="training/arena dependency"):
        build_release(tmp_path)


def test_release_rejects_stale_vaccination_inputs(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    vaccination = json.loads(
        (results / "vaccination.json").read_text(encoding="utf-8")
    )
    vaccination["arena_log_sha256"] = "0" * 64
    _json(results / "vaccination.json", vaccination)

    with pytest.raises(ReleaseInvalid, match="vaccination.json"):
        build_release(tmp_path)


def test_release_rejects_tampered_one_shot_evaluation_receipt(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    attempt_path = results / "evaluation_attempt_v3.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["seed"] += 1
    attempt_path.chmod(0o644)
    _json(attempt_path, attempt)
    attempt_path.chmod(0o444)
    frozen = json.loads((results / "frozen_scoring.json").read_text(encoding="utf-8"))
    frozen["scored_at_seed"] = attempt["seed"]
    frozen["attempt_receipt_sha256"] = sha256_file(
        results / "evaluation_attempt_v3.json"
    )
    _json(results / "frozen_scoring.json", frozen)
    evaluation = json.loads((results / "evaluation_v4_release.json").read_text(encoding="utf-8"))
    evaluation["attempt_receipt_sha256"] = sha256_file(attempt_path)
    for name in ("evaluation_attempt_v3.json", "frozen_scoring.json"):
        evaluation["artifact_sha256"][name] = sha256_file(results / name)
    _json(results / "evaluation_v4_release.json", evaluation)

    with pytest.raises(ReleaseInvalid, match="one-shot receipt"):
        build_release(tmp_path)


def test_release_rejects_writable_active_evaluation_receipt(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    (results / "evaluation_attempt_v3.json").chmod(0o644)

    with pytest.raises(ReleaseInvalid, match="mode 0444"):
        build_release(tmp_path)


def test_release_rejects_stale_terminal_receipt_binding(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    release_path = results / "evaluation_v4_release.json"
    evaluation = json.loads(release_path.read_text(encoding="utf-8"))
    evaluation["attempt_receipt_sha256"] = "0" * 64
    _json(release_path, evaluation)

    with pytest.raises(ReleaseInvalid, match="evaluation gates"):
        build_release(tmp_path)


def test_release_rejects_failed_generalization_contract(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    generalization = json.loads((results / "generalization.json").read_text(encoding="utf-8"))
    generalization["expected_order_holds"] = False
    _json(results / "generalization.json", generalization)
    evaluation = json.loads((results / "evaluation_v4_release.json").read_text(encoding="utf-8"))
    evaluation["artifact_sha256"]["generalization.json"] = sha256_file(
        results / "generalization.json"
    )
    _json(results / "evaluation_v4_release.json", evaluation)

    with pytest.raises(ReleaseInvalid, match="evaluation gates"):
        build_release(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("latency_benchmark_scope", "A+B+C+D+E"),
        ("protected_partition_inference_passes", 2),
        ("protected_partition_inference_passes", True),
    ),
)
def test_release_rejects_noncanonical_latency_custody_even_when_reports_match(
    tmp_path: Path, field: str, value: object
) -> None:
    results = _release_tree(tmp_path)
    metrics_path = results / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["latency"][field] = value
    _json(metrics_path, metrics)
    frozen_path = results / "frozen_scoring.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen[field] = value
    _json(frozen_path, frozen)
    release_path = results / "evaluation_v4_release.json"
    evaluation = json.loads(release_path.read_text(encoding="utf-8"))
    evaluation["artifact_sha256"]["metrics.json"] = sha256_file(metrics_path)
    evaluation["artifact_sha256"]["frozen_scoring.json"] = sha256_file(frozen_path)
    _json(release_path, evaluation)

    with pytest.raises(ReleaseInvalid, match="evaluation gates"):
        build_release(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("latency_benchmark_scope", "C+D+E"),
        ("protected_partition_inference_passes", 2),
    ),
)
def test_release_rejects_metrics_frozen_latency_custody_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    results = _release_tree(tmp_path)
    frozen_path = results / "frozen_scoring.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen[field] = value
    _json(frozen_path, frozen)
    release_path = results / "evaluation_v4_release.json"
    evaluation = json.loads(release_path.read_text(encoding="utf-8"))
    evaluation["artifact_sha256"]["frozen_scoring.json"] = sha256_file(frozen_path)
    _json(release_path, evaluation)

    with pytest.raises(ReleaseInvalid, match="evaluation gates"):
        build_release(tmp_path)


def test_release_requires_passing_v4_terminal(tmp_path: Path) -> None:
    results = _release_tree(tmp_path)
    release_path = results / "evaluation_v4_release.json"
    evaluation = json.loads(release_path.read_text(encoding="utf-8"))
    evaluation["status"] = "failed"
    _json(release_path, evaluation)

    with pytest.raises(ReleaseInvalid, match="status=passed"):
        build_release(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("independent_holdout", True), ("prior_families_disclosed", False)),
)
def test_release_rejects_overclaimed_v4_scope(
    tmp_path: Path, field: str, value: object
) -> None:
    results = _release_tree(tmp_path)
    generalization_path = results / "generalization.json"
    generalization = json.loads(generalization_path.read_text(encoding="utf-8"))
    generalization["evaluation_scope"][field] = value
    _json(generalization_path, generalization)
    release_path = results / "evaluation_v4_release.json"
    evaluation = json.loads(release_path.read_text(encoding="utf-8"))
    evaluation["evaluation_scope"][field] = value
    evaluation["artifact_sha256"]["generalization.json"] = sha256_file(
        generalization_path
    )
    _json(release_path, evaluation)

    with pytest.raises(ReleaseInvalid, match="evaluation gates"):
        build_release(tmp_path)


def test_release_rejects_model_cache_symlink_escape(tmp_path: Path) -> None:
    _release_tree(tmp_path)
    model_root = (
        tmp_path
        / "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    )
    (model_root / "snapshots/revision/escape").symlink_to(tmp_path / "requirements.lock")

    with pytest.raises(ReleaseInvalid, match="embedding model file is missing or unsafe"):
        build_release(tmp_path)


def test_all_dry_run_is_ordered_and_never_opens_entity_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["make", "--no-print-directory", "-n", "all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    ordered = (
        "local_files_only=True",
        "-m atlas.render",
        "scripts/download_reference.py",
        "-m sim.fidelity",
        "-m sim.population --events 50000 --seed 20260829 --evaluation-preregistration results/evaluation_v4_preregistration.json --output data/events.jsonl",
        "-m defend.train",
        "-m loop.arena --rerun",
        "-m defend.evaluate",
        "-m defend.vaccination",
        "-m sim.external_challenge",
        "npm run build",
        "scripts/release.py",
        "-m pytest -q",
        "npm run test:e2e:real",
    )
    positions = [output.index(command) for command in ordered]
    assert positions == sorted(positions)
    assert output.rindex("scripts/release.py --verify") > output.index("npm run test:e2e:real")
    assert "download_retail_reference.py" not in output
    assert "download_retail_ii_reference.py" not in output
    assert "-m sim.entity_fidelity" not in output
    assert "--behavior-profile" not in output
    assert "entity_fidelity_v4 readiness" not in output
    assert "entity_fidelity_v4 sealed" not in output


def test_clean_preserves_lifecycle_and_read_only_evidence(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    lifecycle = results / "entity_fidelity_future_attempt.json"
    lifecycle.write_text("{}\n", encoding="utf-8")
    read_only = results / "future_immutable_evidence.json"
    read_only.write_text("{}\n", encoding="utf-8")
    read_only.chmod(0o444)
    rebuildable = results / "metrics.json"
    rebuildable.write_text("{}\n", encoding="utf-8")

    subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-f",
            str(Path(__file__).resolve().parents[1] / "Makefile"),
            "clean",
            f"PYTHON={sys.executable}",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert lifecycle.is_file()
    assert read_only.is_file()
    assert not rebuildable.exists()


def test_container_context_contains_only_bound_runtime_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    ignore_lines = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    allowed_results = {
        line.removeprefix("!results/")
        for line in ignore_lines
        if line.startswith("!results/")
    }
    assert allowed_results == set(
        ARENA_ARTIFACTS | EVALUATION_ARTIFACTS | RUNTIME_ARTIFACTS
    ) | {MANIFEST, PREREGISTERED_RELEASE_SCRIPT, RELEASE_POLICY_AMENDMENT}
    assert "!data/events.manifest.json" in ignore_lines
    assert "!data/market_prices.csv" in ignore_lines
    assert "app/frontend/dist" not in ignore_lines
    assert "sim/attacks/heldout" in ignore_lines
    assert (
        "!data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/**"
        in ignore_lines
    )

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "COPY --chown=appuser:appuser contracts/event.schema.json "
        "./contracts/event.schema.json"
        in dockerfile
    )
    assert (
        "COPY --chown=appuser:appuser data/events.manifest.json "
        "data/market_prices.csv ./data/"
        in dockerfile
    )
    assert (
        "COPY --chown=appuser:appuser "
        "data/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        in dockerfile
    )
    assert (
        "COPY --chown=appuser:appuser app/frontend/dist ./app/frontend/dist"
        in dockerfile
    )
    assert (
        "COPY --chown=appuser:appuser app/frontend/package.json "
        "app/frontend/package-lock.json ./app/frontend/"
        in dockerfile
    )
    assert "FROM node:" not in dockerfile
    assert "npm" not in dockerfile
    assert "--from=frontend" not in dockerfile
    assert dockerfile.count("FROM ") == dockerfile.count("@sha256:")
    assert "sim/attacks/heldout" not in dockerfile
    assert "SentenceTransformer" not in dockerfile
    assert "install -d -o root -g root -m 0755 /run/ai-defense" in dockerfile
    assert "/run/ai-defense/release-ed25519.pub.pem" in dockerfile

    smoke = (root / "scripts/container_smoke.sh").read_text(encoding="utf-8")
    assert "AI_DEFENSE_RELEASE_PUBLIC_KEY must name an external absolute" in smoke
    assert "--public-key \"$public_key\"" in smoke
    assert "AI_DEFENSE_RELEASE_PUBLIC_KEY=$container_public_key" in smoke
    assert "dst=$container_public_key,readonly" in smoke


def test_container_smoke_rejects_missing_public_key_before_docker() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop(PUBLIC_KEY_ENV, None)

    completed = subprocess.run(
        ["sh", "scripts/container_smoke.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        "AI_DEFENSE_RELEASE_PUBLIC_KEY must name an external absolute"
        in completed.stderr
    )
