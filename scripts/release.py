"""Build or verify the immutable runtime release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defend.harness import generator_input_tree_hash, source_tree_hash
from sim.mandate_twin import (
    SYNTHETIC_KEY_SCOPE,
    key_registry_sha256,
    observed_event_key_registry,
    validate_synthetic_key_registry,
)
from app.backend.artifacts import (
    CustodyInvalid,
    DEFAULT_PRIVATE_KEY,
    generate_signing_key,
    sign_release,
    verify_release_custody,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = "results"
MANIFEST = "release_manifest.json"
RELEASE_POLICY_AMENDMENT = "evaluation_v4_release_policy_amendment.json"
PREREGISTERED_RELEASE_SCRIPT = "evaluation_v4_preregistered_scripts_release.py"
EVIDENCE_KEYS = (
    "dataset_sha256",
    "split_plan_sha256",
    "source_tree_sha256",
    "events_file_sha256",
    "events_manifest_sha256",
    "behavior_profile_sha256",
    "evaluation_preregistration_sha256",
    "key_registry_sha256",
)
GENERALIZATION_RECALL_KEYS = (
    "known_recall",
    "adaptive_recall",
    "novel_composition_recall",
    "heldout_family_recall",
)
ARENA_ARTIFACTS = frozenset(
    {
        "model.pkl",
        "arena_survivors.jsonl",
        "arena_log.json",
        "phylogeny.json",
        "asr_curve.png",
    }
)
EVALUATION_ARTIFACTS = frozenset(
    {
        "metrics.json",
        "generalization.json",
        "precedents.json",
        "shuffle_control.json",
        "scored_events.jsonl",
        "pr_curve.png",
        "attack_recall_heatmap.png",
        "evaluation_attempt_v3.json",
        "frozen_scoring.json",
    }
)
RUNTIME_ARTIFACTS = frozenset(
    {
        "embeddings.npz",
        "precedent.index",
        "precedent_history.json",
        "atlas_matrix.json",
        "vaccination.json",
        "external_attack_challenge.json",
        "agentdojo_banking_official.json",
        "evaluation_attempt_v1_consumed_failed.json",
        "evaluation_v2_preregistration.json",
        "evaluation_v2_arena_stop_fail.json",
        "evaluation_v3_preregistration.json",
        "evaluation_attempt_v2.json",
        "evaluation_release.json",
        "evaluation_v4_preregistration.json",
        "evaluation_v4_release.json",
        "fidelity.json",
        "entity_fidelity_v4_development_failed.json",
        "entity_fidelity_mrs2_preregistration.json",
        "entity_fidelity_mrs2_development_attempt.json",
        "entity_fidelity_mrs2_development.json",
        "training_manifest.json",
        "shuffle_control_protocol_v2.json",
        "shuffle_null_diagnosis_post_crossfit.json",
        "arena_release.json",
    }
)
RUNTIME_PYTHON_DIRECTORIES = ("app/backend", "defend", "loop")
RUNTIME_CONTENT_DIRECTORIES = ("app/frontend/dist",)
RUNTIME_DATA_FILES = ("data/events.manifest.json", "data/market_prices.csv")
RUNTIME_MODEL_DIRECTORY = (
    "data/models/huggingface/hub/"
    "models--sentence-transformers--all-MiniLM-L6-v2"
)
RUNTIME_FILES = (
    "app/__init__.py",
    "app/backend/artifacts.py",
    "contracts/event.schema.json",
    "scripts/release.py",
    "sim/__init__.py",
    "sim/population.py",
    "sim/mandate_twin.py",
)
DEPENDENCY_PACKAGING_FILES = (
    ".dockerignore",
    "Dockerfile",
    "app/frontend/package-lock.json",
    "app/frontend/package.json",
    "requirements.lock",
    "requirements.txt",
)
ENTITY_FIDELITY_LIMITATION_SHA256 = {
    "entity_fidelity_v4_development_failed.json": "81ca82b38ce4194938b4a35be5c5397a4b9e7b67121a249a5057861722d8b312",
    "entity_fidelity_mrs2_preregistration.json": "906467fa115914e5102ecd5924c2c5cebc299864bd79ba34f4ce00db6e544b94",
    "entity_fidelity_mrs2_development_attempt.json": "188a354849afd94bbad7a723f413f3c3acc09c438cffb060262aaa13e60f7c1c",
    "entity_fidelity_mrs2_development.json": "dddb920b50b89f0fe026c5a4f2a197ad236ec561fb61054190313a6889517453",
}
AGENTDOJO_OFFICIAL_SHA256 = "aa7d1ffbbbaa7d7e8c94be221f488ebae732cd3d18869354115c625cea055ced"
EVALUATION_V1_FAILURE_SHA256 = "cd7334f4cedc2b8f09233ef743f342c3135f3854afdc830c6eb8c24bb6c89c57"
EVALUATION_V2_PREREGISTRATION_SHA256 = "418fe884214fa67dc9bb62122b9312eebfc9467b1b0f919903da0da4b8f29d2d"
EVALUATION_V2_ARENA_STOP_SHA256 = "fe54fa15911a7612b0debd6301d4f327356175fdeb942c5a78b5b6e1a34b0104"
EVALUATION_V3_PREREGISTRATION_SHA256 = "08b774b653c8edc246443cf8ccefe4d299f07481bd3b23ce1b386befa82f2401"
EVALUATION_V2_ATTEMPT_SHA256 = "6e3b2e615f99ff44f99f715ac46725eff550a89cecada0ef5291c7bb5505f5d5"
EVALUATION_V3_FAILURE_RELEASE_SHA256 = "4c312af72f48ae83bb160593df1913ea49c1ab733a6118c4b56f8586b7cb9c69"
HISTORICAL_EVALUATION_SHA256 = {
    "evaluation_v3_preregistration.json": EVALUATION_V3_PREREGISTRATION_SHA256,
    "evaluation_attempt_v2.json": EVALUATION_V2_ATTEMPT_SHA256,
    "evaluation_release.json": EVALUATION_V3_FAILURE_RELEASE_SHA256,
}
DISCLOSED_VECTOR_IDS = ["GPF-A20", "GPF-C04"]
DISCLOSED_FAMILY_IDS = ["synthetic_identity_bust_out", "ttl_dormancy"]
V4_LATENCY_BENCHMARK_SCOPE = "A+B"
V4_PROTECTED_PARTITION_INFERENCE_PASSES = 1
V4_PREREGISTERED_RELEASE_SCRIPT_SHA256 = (
    "9d1eb4c35de9a97d64990aa8964b53f5286c70bcbce5f19abcc670c8528798a8"
)
V4_RELEASE_POLICY_EVIDENCE_SHA256 = {
    "evaluation_v4_preregistration.json": "c96d62b877f953b7733cb93d83c81c94f73e847854a42a9176aaf4f3b1913c96",
    "evaluation_attempt_v3.json": "13f92eaed3dd8246ea4b381bfb0927f7b9af24dea8e24caa8455a0a97f221b52",
    "evaluation_v4_release.json": "9cd09acb6b96c6dbc0cf286ef0b76e291999deff96bcaba61138b7ee8d81c929",
    "generalization.json": "1cd8e68dffa08f56f811698c21d50083380a70cf41f17c25701e1860ae8082f7",
    "metrics.json": "492748d1a59c3056f1b9f918b450cdaeac3f72f1a02b257b5c767050acbf6754",
    "frozen_scoring.json": "1f0f522e0da0ef9e0ddd84316e53ce5c21a56c509e8e8629b66ac5755d649c1b",
    "model.pkl": "7b167cbe572bba2668ec52d1f387cbf949b112ef750cfeb7a5ca960a79fd934f",
    "shuffle_control.json": "1d95daf9ce66e14a7e40cfae1a814d701f4ea0c90ecdebd4fef50d79f50c8e11",
}
V4_WITHDRAWN_RELEASE_POLICY_CLAUSES = [
    "generalization.expected_order_holds is True",
    "known_recall >= adaptive_recall >= novel_composition_recall >= heldout_family_recall",
]


class ReleaseInvalid(RuntimeError):
    """The on-disk bundle is incomplete, stale, or internally inconsistent."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReleaseInvalid(f"required artifact is missing or unsafe: {path.name}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReleaseInvalid(f"required artifact is unreadable: {path.name}") from error
    return digest.hexdigest()


def _file_tree(
    project_root: Path,
    paths: list[Path],
    label: str,
    *,
    internal_symlink_root: Path | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    symlink_root = internal_symlink_root.resolve() if internal_symlink_root else None
    for path in sorted(set(paths)):
        relative = path.relative_to(project_root).as_posix()
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
                if symlink_root is None or not resolved.is_file():
                    raise ValueError
                resolved.relative_to(symlink_root)
            except (OSError, RuntimeError, ValueError) as error:
                raise ReleaseInvalid(f"{label} file is missing or unsafe: {relative}") from error
        elif not path.is_file():
            raise ReleaseInvalid(f"{label} file is missing or unsafe: {relative}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ReleaseInvalid(f"{label} file is unreadable: {relative}") from error
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    if not hashes:
        raise ReleaseInvalid(f"{label} file tree is empty")
    return {
        "tree_sha256": digest.hexdigest(),
        "file_sha256": hashes,
        "tree_hash_format": "sha256(relative_posix_path\\0file_bytes\\0)",
    }


def _runtime_code_tree(project_root: Path) -> dict[str, Any]:
    paths = [project_root / name for name in RUNTIME_FILES]
    for name in RUNTIME_PYTHON_DIRECTORIES:
        directory = project_root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ReleaseInvalid(f"runtime code directory is missing or unsafe: {name}")
        descendants = list(directory.rglob("*"))
        if any(path.is_symlink() for path in descendants):
            raise ReleaseInvalid(f"runtime code directory contains a symlink: {name}")
        if any(
            path.is_file()
            and path.suffix != ".py"
            and "__pycache__" not in path.relative_to(directory).parts
            for path in descendants
        ):
            raise ReleaseInvalid(f"runtime code directory contains an unbound file: {name}")
        files = [path for path in descendants if path.is_file() and path.suffix == ".py"]
        if not files:
            raise ReleaseInvalid(f"runtime code directory has no Python files: {name}")
        paths.extend(files)
    for name in RUNTIME_CONTENT_DIRECTORIES:
        directory = project_root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ReleaseInvalid(f"served content directory is missing or unsafe: {name}")
        descendants = list(directory.rglob("*"))
        if any(path.is_symlink() for path in descendants):
            raise ReleaseInvalid(f"served content directory contains a symlink: {name}")
        files = [path for path in descendants if path.is_file()]
        if not files:
            raise ReleaseInvalid(f"served content directory is empty: {name}")
        paths.extend(files)
    return _file_tree(project_root, paths, "runtime code")


def _dependency_packaging_tree(project_root: Path) -> dict[str, Any]:
    return _file_tree(
        project_root,
        [project_root / name for name in DEPENDENCY_PACKAGING_FILES],
        "dependency packaging",
    )


def _runtime_data_tree(project_root: Path) -> dict[str, Any]:
    return _file_tree(
        project_root,
        [project_root / name for name in RUNTIME_DATA_FILES],
        "runtime data",
    )


def _embedding_model_tree(project_root: Path) -> dict[str, Any]:
    directory = project_root / RUNTIME_MODEL_DIRECTORY
    if directory.is_symlink() or not directory.is_dir():
        raise ReleaseInvalid(
            f"embedding model directory is missing or unsafe: {RUNTIME_MODEL_DIRECTORY}"
        )
    descendants = list(directory.rglob("*"))
    if any(path.is_symlink() and not path.is_file() for path in descendants):
        raise ReleaseInvalid("embedding model directory contains an unsafe symlink")
    files = [path for path in descendants if path.is_file()]
    return _file_tree(
        project_root,
        files,
        "embedding model",
        internal_symlink_root=directory,
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseInvalid(f"required artifact is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseInvalid(f"required JSON artifact is invalid: {path.name}") from error
    if not isinstance(value, dict):
        raise ReleaseInvalid(f"required JSON artifact is not an object: {path.name}")
    return value


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ReleaseInvalid(f"required evidence field is missing: {'.'.join(path)}")
        value = value[key]
    return value


def _safe_artifact_name(name: Any) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ReleaseInvalid("upstream release contains an unsafe artifact name")
    return name


def _evaluation_scope_claim() -> dict[str, Any]:
    return {
        "claim": "fresh_row_post_remediation_regression",
        "claim_scope": "post_remediation_regression",
        "fresh_rows_only": True,
        "independent_holdout": False,
        "pristine_heldout_family": False,
        "prior_families_disclosed": True,
        "disclosed_vector_ids": DISCLOSED_VECTOR_IDS,
        "disclosed_family_ids": DISCLOSED_FAMILY_IDS,
    }


def _require_historical_evaluation_artifacts(
    results: Path, artifact_sha256: Mapping[str, str]
) -> None:
    for name, expected_sha256 in HISTORICAL_EVALUATION_SHA256.items():
        path = results / name
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as error:
            raise ReleaseInvalid(f"historical evaluation artifact is missing: {name}") from error
        if artifact_sha256.get(name) != expected_sha256 or mode != 0o444:
            raise ReleaseInvalid(
                f"historical evaluation artifact changed or is not mode 0444: {name}"
            )


def _evaluation_preregistration_bindings(project_root: Path) -> dict[str, str]:
    return {
        "sim_population_sha256": sha256_file(project_root / "sim" / "population.py"),
        "sim_mandate_twin_sha256": sha256_file(
            project_root / "sim" / "mandate_twin.py"
        ),
        "event_schema_sha256": sha256_file(
            project_root / "contracts" / "event.schema.json"
        ),
        "generator_input_tree_sha256": generator_input_tree_hash(project_root),
        "defense_source_tree_sha256": source_tree_hash(project_root),
        "scripts_release_sha256": sha256_file(
            project_root / "scripts" / "release.py"
        ),
        "makefile_sha256": sha256_file(project_root / "Makefile"),
        "requirements_lock_sha256": sha256_file(
            project_root / "requirements.lock"
        ),
        "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "embedding_tree_sha256": _embedding_model_tree(project_root)["tree_sha256"],
        "parent_failure_sha256": EVALUATION_V3_FAILURE_RELEASE_SHA256,
    }


def _evaluation_preregistration_payload(project_root: Path) -> dict[str, Any]:
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
            "disclosed_vector_ids": list(DISCLOSED_VECTOR_IDS),
            "disclosed_family_ids": list(DISCLOSED_FAMILY_IDS),
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
        "bindings": _evaluation_preregistration_bindings(project_root),
    }


def _canonical_evaluation_preregistration(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def preregister_evaluation(project_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    results = root / RESULTS
    if results.is_symlink() or not results.is_dir():
        raise ReleaseInvalid("results directory is missing or unsafe")

    historical_sha256 = {
        name: sha256_file(results / name) for name in HISTORICAL_EVALUATION_SHA256
    }
    _require_historical_evaluation_artifacts(results, historical_sha256)

    destination = results / "evaluation_v4_preregistration.json"
    current_state = (
        destination,
        results / "evaluation_attempt_v3.json",
        results / "evaluation_v4_release.json",
    )
    present = [path.name for path in current_state if path.exists() or path.is_symlink()]
    if present:
        raise ReleaseInvalid(
            "refusing to overwrite or extend partial V4 evaluation state: "
            + ", ".join(present)
        )

    payload = _evaluation_preregistration_payload(root)
    encoded = _canonical_evaluation_preregistration(payload)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReleaseInvalid("evaluation preregistration requires O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o444)
    except OSError as error:
        raise ReleaseInvalid(
            "refusing to overwrite or extend partial V4 evaluation state"
        ) from error
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("evaluation preregistration write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except OSError as error:
        raise ReleaseInvalid(
            "evaluation preregistration write failed; partial state requires audit"
        ) from error
    finally:
        os.close(descriptor)

    try:
        directory = os.open(
            results, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise ReleaseInvalid(
            "evaluation preregistration directory sync failed; state requires audit"
        ) from error

    artifact_sha256 = sha256_file(destination)
    if (
        stat.S_IMODE(destination.stat().st_mode) != 0o444
        or destination.read_bytes() != encoded
    ):
        raise ReleaseInvalid("created evaluation preregistration failed canonical validation")
    _require_historical_evaluation_artifacts(
        results,
        {name: sha256_file(results / name) for name in HISTORICAL_EVALUATION_SHA256},
    )
    created = _read_json(destination)
    if created != payload:
        raise ReleaseInvalid("created evaluation preregistration failed canonical validation")
    _require_evaluation_preregistration(
        created,
        artifact_sha256=artifact_sha256,
        project_root=root,
    )
    return {
        "status": "created",
        "artifact": destination.name,
        "sha256": artifact_sha256,
        "mode": "0444",
    }


def _observed_generalization_order(
    generalization: Mapping[str, Any],
) -> tuple[dict[str, float], bool]:
    try:
        recalls = {
            key: float(generalization[key]) for key in GENERALIZATION_RECALL_KEYS
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseInvalid("generalization recall evidence is missing or invalid") from error
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in recalls.values()):
        raise ReleaseInvalid("generalization recall evidence is missing or invalid")
    values = list(recalls.values())
    return recalls, all(left >= right for left, right in zip(values, values[1:]))


def _release_policy_amendment_payload(project_root: Path) -> dict[str, Any]:
    results = project_root / RESULTS
    generalization = _read_json(results / "generalization.json")
    recalls, observed_order = _observed_generalization_order(generalization)
    bindings = {
        name: sha256_file(results / name)
        for name in V4_RELEASE_POLICY_EVIDENCE_SHA256
    }
    bindings.update(
        {
            "preregistered_release_script_artifact": PREREGISTERED_RELEASE_SCRIPT,
            "preregistered_release_script_sha256": sha256_file(
                results / PREREGISTERED_RELEASE_SCRIPT
            ),
            "amended_release_script_sha256": sha256_file(
                project_root / "scripts" / "release.py"
            ),
        }
    )
    return {
        "schema_version": 1,
        "artifact": "evaluation_v4_release_policy_amendment",
        "status": "post_evaluation_release_policy_amendment",
        "claim_scope": "post_remediation_regression",
        "independent_holdout": False,
        "created_after_evaluation": True,
        "expected_order_holds": observed_order,
        "observed_generalization_recalls": recalls,
        "withdrawn_release_policy_clauses": V4_WITHDRAWN_RELEASE_POLICY_CLAUSES,
        "retained_release_policy": "all original release gates except the two explicitly withdrawn clauses",
        "shuffle_validator_correction": {
            "classification": "validator correction; not a withdrawn quality gate",
            "historical_prior_v1_status": "FAIL",
            "current_v1_diagnostic_status": "PASS",
            "current_v1_certification_use": "preserved diagnostic only",
            "prospective_v2_gate_required": True,
        },
        "invariants": {
            "evaluation_rescored": False,
            "model_changed": False,
            "detection_threshold_changed": False,
            "evaluation_evidence_changed": False,
            "protected_partition_inference_passes": 1,
        },
        "bindings": bindings,
    }


def _require_release_policy_amendment(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    validate_build_bindings: bool = True,
    require_amendment_file: bool = True,
) -> None:
    results = project_root / RESULTS
    generalization = _read_json(results / "generalization.json")
    metrics = _read_json(results / "metrics.json")
    preregistration = _read_json(results / "evaluation_v4_preregistration.json")
    shuffle_control = _read_json(results / "shuffle_control.json")
    recalls, observed_order = _observed_generalization_order(generalization)
    reported_order = generalization.get("expected_order_holds")
    expected_bindings = {
        name: sha256_file(results / name)
        for name in V4_RELEASE_POLICY_EVIDENCE_SHA256
    }
    expected_bindings.update(
        {
            "preregistered_release_script_artifact": PREREGISTERED_RELEASE_SCRIPT,
            "preregistered_release_script_sha256": sha256_file(
                results / PREREGISTERED_RELEASE_SCRIPT
            ),
            "amended_release_script_sha256": sha256_file(
                project_root / "scripts" / "release.py"
            ),
        }
    )
    current_preregistration_bindings = (
        _evaluation_preregistration_bindings(project_root)
        if validate_build_bindings
        else None
    )
    preregistration_bindings = preregistration.get("bindings")
    immutable_names = [
        "evaluation_v4_preregistration.json",
        "evaluation_attempt_v3.json",
        "evaluation_v4_release.json",
        PREREGISTERED_RELEASE_SCRIPT,
    ]
    if require_amendment_file:
        immutable_names.append(RELEASE_POLICY_AMENDMENT)
    if (
        set(payload)
        != {
            "schema_version",
            "artifact",
            "status",
            "claim_scope",
            "independent_holdout",
            "created_after_evaluation",
            "expected_order_holds",
            "observed_generalization_recalls",
            "withdrawn_release_policy_clauses",
            "retained_release_policy",
            "shuffle_validator_correction",
            "invariants",
            "bindings",
        }
        or payload.get("schema_version") != 1
        or payload.get("artifact") != "evaluation_v4_release_policy_amendment"
        or payload.get("status") != "post_evaluation_release_policy_amendment"
        or payload.get("claim_scope") != "post_remediation_regression"
        or payload.get("independent_holdout") is not False
        or payload.get("created_after_evaluation") is not True
        or type(reported_order) is not bool
        or reported_order != observed_order
        or observed_order is not False
        or payload.get("expected_order_holds") != observed_order
        or payload.get("observed_generalization_recalls") != recalls
        or payload.get("withdrawn_release_policy_clauses")
        != V4_WITHDRAWN_RELEASE_POLICY_CLAUSES
        or payload.get("retained_release_policy")
        != "all original release gates except the two explicitly withdrawn clauses"
        or payload.get("shuffle_validator_correction")
        != {
            "classification": "validator correction; not a withdrawn quality gate",
            "historical_prior_v1_status": "FAIL",
            "current_v1_diagnostic_status": "PASS",
            "current_v1_certification_use": "preserved diagnostic only",
            "prospective_v2_gate_required": True,
        }
        or payload.get("invariants")
        != {
            "evaluation_rescored": False,
            "model_changed": False,
            "detection_threshold_changed": False,
            "evaluation_evidence_changed": False,
            "protected_partition_inference_passes": 1,
        }
        or payload.get("bindings") != expected_bindings
        or expected_bindings != {
            **V4_RELEASE_POLICY_EVIDENCE_SHA256,
            "preregistered_release_script_artifact": PREREGISTERED_RELEASE_SCRIPT,
            "preregistered_release_script_sha256": V4_PREREGISTERED_RELEASE_SCRIPT_SHA256,
            "amended_release_script_sha256": expected_bindings[
                "amended_release_script_sha256"
            ],
        }
        or not isinstance(preregistration_bindings, Mapping)
        or preregistration_bindings.get("scripts_release_sha256")
        != V4_PREREGISTERED_RELEASE_SCRIPT_SHA256
        or (
            current_preregistration_bindings is not None
            and any(
                preregistration_bindings.get(name) != value
                for name, value in current_preregistration_bindings.items()
                if name != "scripts_release_sha256"
            )
        )
        or _nested(preregistration, "evaluation", "detection_threshold") != 0.30
        or metrics.get("detection_threshold") != 0.30
        or _nested(generalization, "evaluation_scope", "independent_holdout") is not False
        or _nested(shuffle_control, "v1_mixed_temporal_a", "status") != "PASS"
        or _nested(shuffle_control, "v1_mixed_temporal_a", "certification_use")
        != "preserved diagnostic only"
        or _nested(shuffle_control, "gate", "passed") is not True
        or any(
            stat.S_IMODE((results / name).stat().st_mode) != 0o444
            for name in immutable_names
        )
    ):
        raise ReleaseInvalid("V4 release-policy amendment is missing, stale, or unsafe")


def write_release_policy_amendment(
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    results = root / RESULTS
    destination = results / RELEASE_POLICY_AMENDMENT
    if results.is_symlink() or not results.is_dir() or destination.exists() or destination.is_symlink():
        raise ReleaseInvalid("refusing to overwrite V4 release-policy amendment")
    payload = _release_policy_amendment_payload(root)
    _require_release_policy_amendment(
        payload, project_root=root, require_amendment_file=False
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReleaseInvalid("release-policy amendment requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("release-policy amendment write made no progress")
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(
            results, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise ReleaseInvalid("release-policy amendment write failed; state requires audit") from error
    _require_release_policy_amendment(_read_json(destination), project_root=root)
    return {
        "status": "created",
        "artifact": destination.name,
        "sha256": sha256_file(destination),
        "mode": "0444",
    }


def _upstream_artifacts(
    results: Path,
    release: Mapping[str, Any],
    required: frozenset[str],
    release_name: str,
) -> dict[str, str]:
    declared = release.get("artifact_sha256")
    if not isinstance(declared, Mapping):
        raise ReleaseInvalid(f"{release_name} has no artifact hash map")
    if set(declared) != required:
        missing = sorted(required - set(declared))
        unexpected = sorted(set(declared) - required)
        if missing:
            raise ReleaseInvalid(
                f"{release_name} is missing artifact hashes: {', '.join(missing)}"
            )
        raise ReleaseInvalid(
            f"{release_name} has unexpected artifact hashes: {', '.join(unexpected)}"
        )
    verified: dict[str, str] = {}
    for raw_name, expected in declared.items():
        name = _safe_artifact_name(raw_name)
        if not _is_sha256(expected):
            raise ReleaseInvalid(f"{release_name} has an invalid SHA-256 for {name}")
        path = results / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ReleaseInvalid(f"{release_name} artifact hash is stale: {name}")
        verified[name] = expected
    return verified


def _require_fidelity_gate(payload: Mapping[str, Any]) -> None:
    try:
        auc = float(payload["auc_mean"])
        threshold = float(payload["threshold_max_auc"])
        events = payload["events_generated"]
        required_events = payload["gate_size_required"]
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseInvalid("fidelity.json is missing numeric gate evidence") from error
    if (
        payload.get("schema_version") != 1
        or payload.get("gate_passed") is not True
        or payload.get("threshold_passed") is not True
        or type(events) is not int
        or type(required_events) is not int
        or events != required_events
        or required_events != 200_000
        or not math.isfinite(auc)
        or not math.isfinite(threshold)
        or threshold != 0.60
        or auc > threshold
        or _nested(payload, "reference", "calibration_evaluation_disjoint") is not True
    ):
        raise ReleaseInvalid("fidelity.json did not pass the required measured 200k gate")


def _require_event_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_sha256: str,
    events_sha256: str,
    evaluation_preregistration_sha256: str,
    registry_sha256: str,
    observed_registry: Mapping[str, str] | None,
) -> None:
    key_registry = payload.get("key_registry")
    try:
        validated_registry = validate_synthetic_key_registry(
            key_registry, payload.get("synthetic_identity_seed")
        )
    except ValueError as error:
        raise ReleaseInvalid("event manifest key registry is invalid") from error
    if (
        payload.get("schema_version") != 1
        or payload.get("events") != 50_000
        or payload.get("seed") != 20260829
        or payload.get("fraud_events") != 275
        or payload.get("fraud_rate") != 0.0055
        or payload.get("benign_adversarial_events") != 1_000
        or payload.get("sha256") != events_sha256
        or "behavior_profile_sha256" not in payload
        or payload["behavior_profile_sha256"] is not None
        or payload.get("synthetic_pii") is not True
        or payload.get("api_calls") != 0
        or payload.get("synthetic_identity_seed")
        != "ai-defense-lab:public-synthetic-identities:v1"
        or payload.get("key_registry_scope") != SYNTHETIC_KEY_SCOPE
        or not validated_registry
        or (
            observed_registry is not None
            and validated_registry != dict(sorted(observed_registry.items()))
        )
        or payload.get("key_registry_sha256") != key_registry_sha256(validated_registry)
        or registry_sha256 != key_registry_sha256(validated_registry)
        or payload.get("evaluation_preregistration_sha256")
        != evaluation_preregistration_sha256
        or not _is_sha256(manifest_sha256)
    ):
        raise ReleaseInvalid("event manifest is stale, profiled, or lacks synthetic provenance")


def _require_evaluation_preregistration(
    payload: Mapping[str, Any],
    *,
    artifact_sha256: str,
    project_root: Path,
    validate_build_bindings: bool = True,
    preregistered_release_script_sha256: str | None = None,
) -> dict[str, Any]:
    lineage = payload.get("lineage")
    generator = payload.get("generator")
    arena = payload.get("arena")
    evaluation = payload.get("evaluation")
    custody = payload.get("custody")
    remediation = payload.get("remediation")
    bindings = payload.get("bindings")
    binding_hash_names = {
        "sim_population_sha256",
        "sim_mandate_twin_sha256",
        "event_schema_sha256",
        "generator_input_tree_sha256",
        "defense_source_tree_sha256",
        "scripts_release_sha256",
        "makefile_sha256",
        "requirements_lock_sha256",
        "embedding_tree_sha256",
        "parent_failure_sha256",
    }
    expected_bindings = None
    if validate_build_bindings:
        expected_bindings = _evaluation_preregistration_bindings(project_root)
        if preregistered_release_script_sha256 is not None:
            expected_bindings["scripts_release_sha256"] = (
                preregistered_release_script_sha256
            )
    bindings_valid = (
        isinstance(bindings, Mapping)
        and set(bindings) == binding_hash_names | {"embedding_revision"}
        and bindings.get("embedding_revision")
        == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        and bindings.get("parent_failure_sha256")
        == EVALUATION_V3_FAILURE_RELEASE_SHA256
        and all(_is_sha256(bindings.get(name)) for name in binding_hash_names)
        and (expected_bindings is None or dict(bindings) == expected_bindings)
    )
    if (
        set(payload)
        != {
            "schema_version",
            "artifact",
            "status",
            "claim_scope",
            "pristine_heldout_family",
            "independent_holdout",
            "fresh_rows_only",
            "lineage",
            "remediation",
            "generator",
            "arena",
            "evaluation",
            "custody",
            "bindings",
        }
        or payload.get("schema_version") != 4
        or payload.get("artifact") != "evaluation_v4_preregistration"
        or payload.get("status") != "predeclared_before_post_remediation_regression"
        or payload.get("claim_scope") != "post_remediation_regression"
        or payload.get("pristine_heldout_family") is not False
        or payload.get("independent_holdout") is not False
        or payload.get("fresh_rows_only") is not True
    ):
        raise ReleaseInvalid("evaluation V4 preregistration identity is invalid")
    if (
        not isinstance(lineage, Mapping)
        or dict(lineage)
        != {
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
        }
        or not isinstance(remediation, Mapping)
        or dict(remediation)
        != {
            "utilisation_current_min": 0.9,
            "utilisation_delta_min": 0.7,
            "velocity_1h_min": 5,
            "threshold_changed": False,
            "detection_evaluator_changed": False,
            "evaluation_seed_changed": False,
        }
        or not isinstance(generator, Mapping)
        or dict(generator)
        != {
            "seed": 20260829,
            "events": 50_000,
            "fraud_rate": 0.0055,
            "benign_adversarial_rate": 0.02,
            "family_definitions_reused": True,
            "fresh_rows_only": True,
        }
        or not isinstance(arena, Mapping)
        or dict(arena)
        != {
            "seed": 20260827,
            "replay_challenge_count": 48,
            "attack_success_threshold": 0.60,
            "training_partitions": ["A", "B"],
            "gate": "final == 0 or final < baseline",
            "zero_residual_is_floor_pass": True,
        }
        or not isinstance(evaluation, Mapping)
        or dict(evaluation)
        != {
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
        }
        or not isinstance(custody, Mapping)
        or dict(custody)
        != {
            "prior_families_disclosed": True,
            "new_rows_not_new_families": True,
            "training_input_physically_mixed": True,
            "v3_evaluation_attempt_created": False,
            "attempt_receipt": "evaluation_attempt_v3.json",
            "retry_after_consumption": False,
            "independent_holdout": False,
        }
        or not bindings_valid
        or not _is_sha256(artifact_sha256)
    ):
        raise ReleaseInvalid("evaluation V4 preregistration contract is stale or incomplete")
    return _evaluation_scope_claim()


def _require_entity_fidelity_limitations(
    payloads: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if {
        name: artifact_sha256.get(name)
        for name in ENTITY_FIDELITY_LIMITATION_SHA256
    } != ENTITY_FIDELITY_LIMITATION_SHA256:
        raise ReleaseInvalid("entity-fidelity STOP_FAIL evidence is not the immutable reviewed set")

    v4 = payloads["entity_fidelity_v4_development_failed.json"]
    preregistration = payloads["entity_fidelity_mrs2_preregistration.json"]
    attempt = payloads["entity_fidelity_mrs2_development_attempt.json"]
    mrs2 = payloads["entity_fidelity_mrs2_development.json"]
    try:
        maximum = float(_nested(mrs2, "gate", "metrics", "orientation_free_max_seed_d"))
        interval = _nested(mrs2, "gate", "metrics", "bootstrap_max_seed_d_ci_95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise TypeError
        low, high = (float(value) for value in interval)
        threshold = float(_nested(mrs2, "gate", "metrics", "threshold_max_d_upper_95"))
    except (TypeError, ValueError) as error:
        raise ReleaseInvalid("MRS2 STOP_FAIL evidence has invalid gate metrics") from error
    if (
        v4.get("schema_version") != 1
        or v4.get("artifact") != "entity_fidelity_v4_calibration_development_failure"
        or v4.get("immutable") is not True
        or v4.get("status") != "STOP_FAIL"
        or v4.get("release_eligible") is not False
        or v4.get("protocol_frozen") is not False
        or v4.get("calibration_only") is not True
        or v4.get("uci502_transactions_accessed") is not False
        or v4.get("diagnostic_transactions_accessed") is not False
        or v4.get("sealed_final_transactions_accessed") is not False
        or preregistration.get("schema_version") != "mrs2"
        or preregistration.get("artifact") != "entity_fidelity_mrs2_preregistration"
        or _nested(preregistration, "one_shot", "retry_permitted") is not False
        or _nested(preregistration, "privacy", "aggregate_only_artifacts") is not True
        or _nested(preregistration, "privacy", "contains_ids") is not False
        or _nested(preregistration, "privacy", "contains_rows") is not False
        or attempt.get("artifact") != "entity_fidelity_mrs2_development_attempt"
        or attempt.get("status") != "CONSUMED_ONE_SHOT"
        or attempt.get("retry_permitted") is not False
        or attempt.get("qualification_accessed") is not False
        or attempt.get("uci_sources_accessed") is not False
        or attempt.get("mrs2_preregistration_sha256")
        != ENTITY_FIDELITY_LIMITATION_SHA256["entity_fidelity_mrs2_preregistration.json"]
        or mrs2.get("schema_version") != "mrs2"
        or mrs2.get("artifact") != "entity_fidelity_mrs2_development"
        or mrs2.get("status") != "STOP_FAIL"
        or mrs2.get("retry_permitted") is not False
        or mrs2.get("qualification_accessed") is not False
        or mrs2.get("qualification_artifact_created") is not False
        or mrs2.get("uci_sources_accessed") is not False
        or mrs2.get("development_attempt_sha256")
        != ENTITY_FIDELITY_LIMITATION_SHA256["entity_fidelity_mrs2_development_attempt.json"]
        or mrs2.get("mrs2_preregistration_sha256")
        != ENTITY_FIDELITY_LIMITATION_SHA256["entity_fidelity_mrs2_preregistration.json"]
        or _nested(mrs2, "gate", "gate_passed") is not False
        or _nested(mrs2, "development_profile", "privacy", "aggregate_only") is not True
        or _nested(mrs2, "development_profile", "privacy", "contains_ids") is not False
        or _nested(mrs2, "development_profile", "privacy", "contains_rows") is not False
        or not all(math.isfinite(value) for value in (maximum, low, high, threshold))
        or not 0.5 <= maximum <= 1.0
        or not 0.5 <= low <= high <= 1.0
        or threshold != 0.65
        or high <= threshold
    ):
        raise ReleaseInvalid("entity-fidelity evidence does not preserve the reviewed STOP_FAIL limitations")
    return {
        "status": "STOP_FAIL",
        "release_eligible": False,
        "runtime_behavior_profile_sha256": None,
        "scope": "The 200k marginal fidelity gate passed; entity-behavior fidelity is not certified.",
        "v4": {
            "status": "STOP_FAIL",
            "artifact_sha256": artifact_sha256["entity_fidelity_v4_development_failed.json"],
            "sealed_or_readiness_scoring_performed": False,
        },
        "mrs2": {
            "status": "STOP_FAIL",
            "artifact_sha256": artifact_sha256["entity_fidelity_mrs2_development.json"],
            "orientation_free_max_seed_d": maximum,
            "bootstrap_max_seed_d_ci_95": [low, high],
            "threshold_max_d_upper_95": threshold,
            "qualification_accessed": False,
            "retry_permitted": False,
        },
    }


def _require_external_challenge(
    payload: Mapping[str, Any],
    model_sha256: str,
    official: Mapping[str, Any],
    official_sha256: str,
    evidence: Mapping[str, str | None],
) -> None:
    non_contamination = _nested(payload, "non_contamination")
    safety = _nested(payload, "safety")
    agentdojo = _nested(payload, "official_agentdojo")
    agentdojo_evidence = _nested(agentdojo, "evidence")
    agentdojo_run = _nested(agentdojo, "run")
    if not all(
        isinstance(value, Mapping)
        for value in (non_contamination, safety, agentdojo, agentdojo_evidence, agentdojo_run)
    ):
        raise ReleaseInvalid("external attack challenge is stale or unsafe")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol") != "sealed_external_attack_challenge"
        or payload.get("model_hash") != model_sha256
        or payload.get("events_manifest_sha256")
        != evidence.get("events_manifest_sha256")
        or payload.get("key_registry_sha256")
        != evidence.get("key_registry_sha256")
        or non_contamination.get("evaluation_only") is not True
        or non_contamination.get("frozen_model_hash_invariant") is not True
        or non_contamination.get("model_hash_before") != model_sha256
        or non_contamination.get("model_hash_after") != model_sha256
        or safety.get("raw_payloads_included") is not False
        or safety.get("synthetic_pii_only") is not True
        or official_sha256 != AGENTDOJO_OFFICIAL_SHA256
        or agentdojo_evidence.get("file") != "results/agentdojo_banking_official.json"
        or agentdojo_evidence.get("artifact_sha256") != official_sha256
        or agentdojo_evidence.get("canonical_sha256_match") is not True
        or agentdojo_evidence.get("structural_consistency_valid") is not True
        or agentdojo_evidence.get("authenticity_verified") is not False
        or agentdojo_evidence.get("authenticity_note")
        != "Require this artifact SHA-256 in the signed release manifest."
        or agentdojo_evidence.get("raw_logs_verified") is not False
        or agentdojo_evidence.get("injection_utility_validation") != "aggregate_only"
        or agentdojo.get("source") != official.get("source")
        or agentdojo.get("configuration") != official.get("configuration")
        or agentdojo.get("configuration_sha256") != official.get("configuration_sha256")
        or agentdojo_run.get("protocol") != official.get("protocol")
        or agentdojo_run.get("run_id") != official.get("run_id")
        or agentdojo_run.get("generated_at") != official.get("generated_at")
        or agentdojo_run.get("official_harness_executed")
        != _nested(official, "fidelity", "official_harness_executed")
        or agentdojo_run.get("official_task_success_evaluators_executed")
        != _nested(official, "fidelity", "official_task_success_evaluators_executed")
        or agentdojo_run.get("attacked_task_pairs")
        != _nested(official, "plan", "attacked_task_pairs")
        or agentdojo.get("utility") != official.get("utility")
        or agentdojo.get("attacker_goal") != official.get("attacker_goal")
    ):
        raise ReleaseInvalid("external attack challenge is stale or unsafe")


def _require_shuffle_control(
    control: Mapping[str, Any],
    protocol: Mapping[str, Any],
    prior_v1: Mapping[str, Any],
    *,
    prior_v1_sha256: str,
    model_sha256: str,
    evidence: Mapping[str, str | None],
    release_policy_amendment: bool = False,
) -> None:
    protocol_id = "temporal-a-has-text-conditional-v2-2026-08-27"
    records = control.get("records")
    if (
        protocol.get("protocol_id") != protocol_id
        or protocol.get("status") != "predeclared_before_execution"
        or _nested(protocol, "prior_v1", "sha256") != prior_v1_sha256
        or _nested(protocol, "prior_v1", "status") != "FAIL"
        or protocol.get("sealed_partitions_opened") != []
        or prior_v1.get("gate", {}).get("passed") is not False
        or control.get("protocol_id") != protocol_id
        or control.get("model_hash") != model_sha256
        or any(key not in control or control[key] != value for key, value in evidence.items())
        or not isinstance(records, list)
        or len(records) != 50
        or control.get("conditional_evaluation_positives") != control.get("evaluation_positives")
        or _nested(control, "protocol", "frozen_partitions_opened") != []
        or (
            not release_policy_amendment
            and _nested(control, "v1_mixed_temporal_a", "status") != "FAIL"
        )
        or (
            release_policy_amendment
            and (
                _nested(control, "v1_mixed_temporal_a", "status") != "PASS"
                or _nested(
                    control, "v1_mixed_temporal_a", "certification_use"
                )
                != "preserved diagnostic only"
            )
        )
        or _nested(control, "gate", "passed") is not True
    ):
        raise ReleaseInvalid("shuffle-control evidence is incomplete, stale, or did not pass")


def _require_pipeline_evidence(
    *,
    training: Mapping[str, Any],
    arena_release: Mapping[str, Any],
    arena_log: Mapping[str, Any],
    frozen_scoring: Mapping[str, Any],
    generalization: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evaluation_release: Mapping[str, Any],
    evaluation_attempt: Mapping[str, Any],
    evaluation_attempt_sha256: str,
    model_sha256: str,
    evidence: Mapping[str, str | None],
    release_policy_amendment: bool = False,
) -> None:
    base_model_sha256 = training.get("model_sha256")
    evaluation_scope = _evaluation_scope_claim()
    hardening_gate = arena_log.get("hardening_gate")
    if not isinstance(hardening_gate, Mapping):
        raise ReleaseInvalid("arena hardening gate evidence is missing")
    baseline = hardening_gate.get("baseline")
    final = hardening_gate.get("final")
    valid_numbers = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (baseline, final)
    )
    expected_outcome = (
        "strict_improvement"
        if valid_numbers and final < baseline
        else "zero_residual_floor"
    )
    if (
        not _is_sha256(base_model_sha256)
        or training.get("training_partitions") != ["A", "B"]
        or arena_release.get("base_model_hash") != base_model_sha256
        or arena_release.get("seed") != 20260827
        or arena_log.get("seed") != 20260827
        or arena_log.get("base_model_hash") != base_model_sha256
        or arena_log.get("final_model_hash") != model_sha256
        or arena_log.get("training_partitions") != ["A", "B"]
        or arena_log.get("input_partitions_materialized") != ["A", "B", "C", "D", "E"]
        or arena_log.get("partitions_used_for_training_or_feature_extraction")
        != ["A", "B"]
        or not valid_numbers
        or hardening_gate.get("operator") != "final == 0 or final < baseline"
        or hardening_gate.get("passed") is not True
        or not (final == 0 or final < baseline)
        or hardening_gate.get("outcome") != expected_outcome
        or any(key not in training or training[key] != value for key, value in evidence.items())
        or any(key not in arena_log or arena_log[key] != value for key, value in evidence.items())
    ):
        raise ReleaseInvalid("training/arena dependency evidence is stale or unsafe")

    if (
        evaluation_attempt.get("schema_version") != 3
        or evaluation_attempt.get("artifact") != "evaluation_attempt_v3"
        or evaluation_attempt.get("status") != "consumed_one_shot"
        or evaluation_attempt.get("claim_scope") != "post_remediation_regression"
        or evaluation_attempt.get("pristine_heldout_family") is not False
        or evaluation_attempt.get("independent_holdout") is not False
        or evaluation_attempt.get("fresh_rows_only") is not True
        or evaluation_attempt.get("prior_families_disclosed") is not True
        or evaluation_attempt.get("disclosed_vector_ids") != DISCLOSED_VECTOR_IDS
        or evaluation_attempt.get("disclosed_family_ids") != DISCLOSED_FAMILY_IDS
        or type(evaluation_attempt.get("seed")) is not int
        or evaluation_attempt.get("seed") != 20260827
        or evaluation_attempt.get("seed") != frozen_scoring.get("scored_at_seed")
        or evaluation_attempt.get("partitions_to_score_once") != ["C", "D", "E"]
        or evaluation_attempt.get("scoring_counts") != frozen_scoring.get("scoring_counts")
        or frozen_scoring.get("attempt_receipt") != "evaluation_attempt_v3.json"
        or frozen_scoring.get("attempt_receipt_sha256") != evaluation_attempt_sha256
        or any(
            key not in evaluation_attempt or evaluation_attempt[key] != value
            for key, value in {"model_hash": model_sha256, **evidence}.items()
        )
    ):
        raise ReleaseInvalid("evaluation one-shot receipt is stale or unsafe")

    if (
        frozen_scoring.get("model_hash") != model_sha256
        or frozen_scoring.get("partitions_scored_once") != ["C", "D", "E"]
        or frozen_scoring.get("weights_unchanged") is not True
        or any(
            key not in frozen_scoring or frozen_scoring[key] != value
            for key, value in evidence.items()
        )
    ):
        raise ReleaseInvalid("frozen C/D/E scoring evidence is stale or unsafe")

    try:
        recalls = [float(generalization[key]) for key in GENERALIZATION_RECALL_KEYS]
        latency_p99 = float(_nested(metrics, "latency", "p99_ms"))
        latency_samples = int(_nested(metrics, "latency", "sample_count"))
        latency_benchmark_scope = _nested(
            metrics, "latency", "latency_benchmark_scope"
        )
        protected_partition_inference_passes = _nested(
            metrics, "latency", "protected_partition_inference_passes"
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseInvalid("evaluation gate evidence is missing or invalid") from error
    frozen_latency_benchmark_scope = frozen_scoring.get("latency_benchmark_scope")
    frozen_protected_partition_inference_passes = frozen_scoring.get(
        "protected_partition_inference_passes"
    )
    checks = generalization.get("contamination_checks")
    required_checks = (
        "shuffled_label_gate",
        "temporal_split",
        "import_firewall",
        "frozen_hash_invariant",
    )
    _, observed_order = _observed_generalization_order(generalization)
    reported_order = generalization.get("expected_order_holds")
    order_valid = (
        type(reported_order) is bool
        and reported_order == observed_order
        and (release_policy_amendment or observed_order)
    )
    if (
        generalization.get("model_hash") != model_sha256
        or generalization.get("headline") != "heldout_family_recall"
        or not order_valid
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in recalls)
        or not isinstance(checks, Mapping)
        or any(checks.get(name) != "PASS" for name in required_checks)
        or generalization.get("evaluation_scope") != evaluation_scope
        or evaluation_release.get("schema_version") != 4
        or evaluation_release.get("artifact") != "evaluation_v4_release"
        or evaluation_release.get("status") != "passed"
        or evaluation_release.get("evaluation_scope") != evaluation_scope
        or evaluation_release.get("attempt_receipt") != "evaluation_attempt_v3.json"
        or evaluation_release.get("attempt_receipt_sha256")
        != evaluation_attempt_sha256
        or not math.isfinite(latency_p99)
        or latency_p99 >= 100
        or latency_samples <= 0
        or latency_benchmark_scope != V4_LATENCY_BENCHMARK_SCOPE
        or frozen_latency_benchmark_scope != latency_benchmark_scope
        or type(protected_partition_inference_passes) is not int
        or protected_partition_inference_passes
        != V4_PROTECTED_PARTITION_INFERENCE_PASSES
        or type(frozen_protected_partition_inference_passes) is not int
        or frozen_protected_partition_inference_passes
        != protected_partition_inference_passes
    ):
        raise ReleaseInvalid("evaluation gates did not pass the production release contract")


def _merge_hashes(destination: dict[str, str], source: Mapping[str, str]) -> None:
    for name, digest in source.items():
        if name in destination and destination[name] != digest:
            raise ReleaseInvalid(f"upstream releases disagree on artifact hash: {name}")
        destination[name] = digest


def _collect(
    project_root: Path,
    *,
    validate_source_events: bool = False,
    trusted_preregistration_sha256: str | None = None,
    release_policy_amendment: bool = False,
) -> dict[str, Any]:
    results = project_root / RESULTS
    arena_path = results / "arena_release.json"
    evaluation_path = results / "evaluation_v4_release.json"
    arena = _read_json(arena_path)
    evaluation = _read_json(evaluation_path)
    if arena.get("status") != "passed" or evaluation.get("status") != "passed":
        raise ReleaseInvalid("arena and evaluation releases must both have status=passed")

    model_path = results / "model.pkl"
    if not model_path.is_file():
        raise ReleaseInvalid("required artifact is missing: model.pkl")
    model_sha256 = sha256_file(model_path)
    if arena.get("final_model_hash") != model_sha256 or evaluation.get("model_hash") != model_sha256:
        raise ReleaseInvalid("arena/evaluation model evidence does not match current model.pkl")

    evidence: dict[str, str | None] = {}
    for key in EVIDENCE_KEYS:
        arena_value = arena.get(key)
        evaluation_value = evaluation.get(key)
        valid = arena_value is None if key == "behavior_profile_sha256" else _is_sha256(arena_value)
        if key not in arena or key not in evaluation or not valid or arena_value != evaluation_value:
            raise ReleaseInvalid(f"arena/evaluation evidence disagrees: {key}")
        evidence[key] = arena_value
    try:
        current_source_sha256 = source_tree_hash(project_root)
    except OSError as error:
        raise ReleaseInvalid("current source tree is incomplete or unreadable") from error
    if evidence["source_tree_sha256"] != current_source_sha256:
        raise ReleaseInvalid("arena/evaluation source-tree evidence is stale")
    event_manifest_path = project_root / "data" / "events.manifest.json"
    if sha256_file(event_manifest_path) != evidence["events_manifest_sha256"]:
        raise ReleaseInvalid("event manifest SHA-256 does not match pipeline evidence")
    observed_registry = None
    if validate_source_events:
        events_path = project_root / "data" / "events.jsonl"
        try:
            if sha256_file(events_path) != evidence["events_file_sha256"]:
                raise ReleaseInvalid("event file SHA-256 does not match pipeline evidence")
            with events_path.open(encoding="utf-8") as handle:
                observed_registry = observed_event_key_registry(
                    json.loads(line) for line in handle if line.strip()
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ReleaseInvalid("event file signer registry is invalid") from error
    _require_event_manifest(
        _read_json(event_manifest_path),
        manifest_sha256=evidence["events_manifest_sha256"],
        events_sha256=evidence["events_file_sha256"],
        evaluation_preregistration_sha256=evidence[
            "evaluation_preregistration_sha256"
        ],
        registry_sha256=evidence["key_registry_sha256"],
        observed_registry=observed_registry,
    )

    artifacts: dict[str, str] = {}
    _merge_hashes(artifacts, _upstream_artifacts(results, arena, ARENA_ARTIFACTS, arena_path.name))
    _merge_hashes(
        artifacts,
        _upstream_artifacts(results, evaluation, EVALUATION_ARTIFACTS, evaluation_path.name),
    )
    for name in sorted(RUNTIME_ARTIFACTS):
        path = results / name
        if not path.is_file():
            raise ReleaseInvalid(f"required artifact is missing: {name}")
        artifacts[name] = sha256_file(path)
    if release_policy_amendment:
        for name in (PREREGISTERED_RELEASE_SCRIPT, RELEASE_POLICY_AMENDMENT):
            artifacts[name] = sha256_file(results / name)

    arena_log = _read_json(results / "arena_log.json")
    frozen_scoring = _read_json(results / "frozen_scoring.json")
    evaluation_attempt = _read_json(results / "evaluation_attempt_v3.json")
    generalization = _read_json(results / "generalization.json")
    metrics = _read_json(results / "metrics.json")
    training = _read_json(results / "training_manifest.json")
    preregistration_sha256 = artifacts["evaluation_v4_preregistration.json"]
    if any(
        stat.S_IMODE((results / name).stat().st_mode) != 0o444
        for name in (
            "evaluation_attempt_v1_consumed_failed.json",
            "evaluation_v2_preregistration.json",
            "evaluation_v2_arena_stop_fail.json",
            "evaluation_v3_preregistration.json",
            "evaluation_attempt_v2.json",
            "evaluation_release.json",
            "evaluation_v4_preregistration.json",
            "evaluation_attempt_v3.json",
        )
    ):
        raise ReleaseInvalid("evaluation lineage artifacts must be mode 0444")
    if (
        evidence["evaluation_preregistration_sha256"] != preregistration_sha256
        or trusted_preregistration_sha256 is not None
        and trusted_preregistration_sha256 != preregistration_sha256
    ):
        raise ReleaseInvalid("evaluation preregistration does not match pipeline evidence")
    evaluation_scope = _require_evaluation_preregistration(
        _read_json(results / "evaluation_v4_preregistration.json"),
        artifact_sha256=preregistration_sha256,
        project_root=project_root,
        validate_build_bindings=trusted_preregistration_sha256 is None,
        preregistered_release_script_sha256=(
            V4_PREREGISTERED_RELEASE_SCRIPT_SHA256
            if release_policy_amendment
            else None
        ),
    )
    amendment = None
    if release_policy_amendment:
        amendment = _read_json(results / RELEASE_POLICY_AMENDMENT)
        _require_release_policy_amendment(
            amendment,
            project_root=project_root,
            validate_build_bindings=trusted_preregistration_sha256 is None,
        )
    if artifacts["evaluation_attempt_v1_consumed_failed.json"] != EVALUATION_V1_FAILURE_SHA256:
        raise ReleaseInvalid("prior consumed evaluation failure evidence changed")
    if artifacts["evaluation_v2_preregistration.json"] != EVALUATION_V2_PREREGISTRATION_SHA256:
        raise ReleaseInvalid("prior evaluation V2 preregistration changed")
    if artifacts["evaluation_v2_arena_stop_fail.json"] != EVALUATION_V2_ARENA_STOP_SHA256:
        raise ReleaseInvalid("prior evaluation V2 arena STOP evidence changed")
    _require_historical_evaluation_artifacts(results, artifacts)
    _require_fidelity_gate(_read_json(results / "fidelity.json"))
    entity_limitations = _require_entity_fidelity_limitations(
        {
            name: _read_json(results / name)
            for name in ENTITY_FIDELITY_LIMITATION_SHA256
        },
        artifacts,
    )
    _require_shuffle_control(
        _read_json(results / "shuffle_control.json"),
        _read_json(results / "shuffle_control_protocol_v2.json"),
        _read_json(results / "shuffle_null_diagnosis_post_crossfit.json"),
        prior_v1_sha256=artifacts["shuffle_null_diagnosis_post_crossfit.json"],
        model_sha256=model_sha256,
        evidence=evidence,
        release_policy_amendment=release_policy_amendment,
    )
    _require_external_challenge(
        _read_json(results / "external_attack_challenge.json"),
        model_sha256,
        _read_json(results / "agentdojo_banking_official.json"),
        artifacts["agentdojo_banking_official.json"],
        evidence,
    )
    _require_pipeline_evidence(
        training=training,
        arena_release=arena,
        arena_log=arena_log,
        frozen_scoring=frozen_scoring,
        generalization=generalization,
        metrics=metrics,
        evaluation_release=evaluation,
        evaluation_attempt=evaluation_attempt,
        evaluation_attempt_sha256=artifacts["evaluation_attempt_v3.json"],
        model_sha256=model_sha256,
        evidence=evidence,
        release_policy_amendment=release_policy_amendment,
    )
    vaccination = _read_json(results / "vaccination.json")
    if (
        vaccination.get("live_model_unchanged") is not True
        or vaccination.get("live_model_sha256") != model_sha256
        or vaccination.get("seed") != 20260827
        or vaccination.get("arena_log_sha256") != artifacts["arena_log.json"]
        or vaccination.get("arena_survivors_sha256")
        != artifacts["arena_survivors.jsonl"]
        or vaccination.get("arena_release_sha256") != artifacts["arena_release.json"]
        or vaccination.get("events_manifest_sha256")
        != evidence["events_manifest_sha256"]
        or vaccination.get("key_registry_sha256")
        != evidence["key_registry_sha256"]
    ):
        raise ReleaseInvalid("vaccination.json does not preserve the frozen live model")

    limitations: dict[str, Any] = {
        "entity_behavior_fidelity": entity_limitations,
        "evaluation_scope": evaluation_scope,
    }
    if amendment is not None:
        limitations["evaluation_release_policy_amendment"] = {
            "artifact": RELEASE_POLICY_AMENDMENT,
            "sha256": artifacts[RELEASE_POLICY_AMENDMENT],
            "status": amendment["status"],
            "expected_order_holds": amendment["expected_order_holds"],
            "observed_generalization_recalls": amendment[
                "observed_generalization_recalls"
            ],
            "withdrawn_release_policy_clauses": amendment[
                "withdrawn_release_policy_clauses"
            ],
            "retained_release_policy": amendment["retained_release_policy"],
            "shuffle_validator_correction": amendment[
                "shuffle_validator_correction"
            ],
            "independent_holdout": False,
        }
    return {
        "schema_version": 6,
        "status": "passed",
        "model_sha256": model_sha256,
        "evidence": evidence,
        "known_limitations": limitations,
        "runtime_code": _runtime_code_tree(project_root),
        "runtime_data": _runtime_data_tree(project_root),
        "embedding_model": _embedding_model_tree(project_root),
        "dependency_packaging": _dependency_packaging_tree(project_root),
        "required_artifact_sha256": dict(sorted(artifacts.items())),
    }


def build_release(
    project_root: str | Path = ROOT,
    private_key_path: str | Path | None = None,
    *,
    release_policy_amendment: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    amendment_path = root / RESULTS / RELEASE_POLICY_AMENDMENT
    if (
        not release_policy_amendment
        and (amendment_path.exists() or amendment_path.is_symlink())
    ):
        raise ReleaseInvalid(
            "V4 release-policy amendment exists; explicit amendment signing is required"
        )
    try:
        payload = sign_release(
            _collect(
                root,
                validate_source_events=not release_policy_amendment,
                release_policy_amendment=release_policy_amendment,
            ),
            root,
            private_key_path,
        )
    except CustodyInvalid as error:
        raise ReleaseInvalid(f"release signing failed: {error}") from error
    results = root / RESULTS
    results.mkdir(parents=True, exist_ok=True)
    destination = results / MANIFEST
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=results, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify_release(
    project_root: str | Path = ROOT,
    public_key_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = root / RESULTS / MANIFEST
    manifest = _read_json(path)
    try:
        payload = verify_release_custody(
            manifest, public_key_path, project_root=root
        )
    except CustodyInvalid as error:
        raise ReleaseInvalid(f"release custody verification failed: {error}") from error
    signed_artifacts = payload.get("required_artifact_sha256")
    trusted_preregistration_sha256 = (
        signed_artifacts.get("evaluation_v4_preregistration.json")
        if isinstance(signed_artifacts, Mapping)
        else None
    )
    if not _is_sha256(trusted_preregistration_sha256):
        raise ReleaseInvalid("signed release has no valid evaluation preregistration hash")
    has_amendment = (
        isinstance(signed_artifacts, Mapping)
        and RELEASE_POLICY_AMENDMENT in signed_artifacts
    )
    has_snapshot = (
        isinstance(signed_artifacts, Mapping)
        and PREREGISTERED_RELEASE_SCRIPT in signed_artifacts
    )
    if has_amendment != has_snapshot:
        raise ReleaseInvalid("signed release has incomplete policy-amendment custody")
    expected = _collect(
        root,
        validate_source_events=False,
        trusted_preregistration_sha256=trusted_preregistration_sha256,
        release_policy_amendment=has_amendment,
    )
    if payload != expected:
        raise ReleaseInvalid("release_manifest.json is stale or does not match the required bundle")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true", help="verify only; never write artifacts")
    parser.add_argument(
        "--preregister-evaluation",
        action="store_true",
        help="create the immutable V4 evaluation preregistration and nothing else",
    )
    parser.add_argument(
        "--write-release-policy-amendment",
        action="store_true",
        help="create the immutable V4 post-evaluation release-policy amendment only",
    )
    parser.add_argument(
        "--release-policy-amendment",
        action="store_true",
        help="sign using the immutable V4 release-policy amendment",
    )
    parser.add_argument(
        "--generate-key",
        nargs="?",
        const=DEFAULT_PRIVATE_KEY,
        type=Path,
        metavar="PRIVATE_KEY_PATH",
        help=f"generate an external Ed25519 key pair (default: {DEFAULT_PRIVATE_KEY})",
    )
    parser.add_argument("--private-key", type=Path, help="external Ed25519 private key for signing")
    parser.add_argument("--public-key", type=Path, help="configured Ed25519 public key")
    args = parser.parse_args()
    if args.write_release_policy_amendment:
        if (
            args.verify
            or args.preregister_evaluation
            or args.release_policy_amendment
            or args.generate_key is not None
            or args.private_key
            or args.public_key
        ):
            parser.error(
                "--write-release-policy-amendment cannot be combined with other modes"
            )
        print(json.dumps(write_release_policy_amendment(args.root), sort_keys=True))
        return
    if args.preregister_evaluation:
        if (
            args.verify
            or args.release_policy_amendment
            or args.generate_key is not None
            or args.private_key
            or args.public_key
        ):
            parser.error(
                "--preregister-evaluation cannot be combined with signing or verification options"
            )
        print(json.dumps(preregister_evaluation(args.root), sort_keys=True))
        return
    if args.generate_key is not None:
        if args.verify or args.release_policy_amendment or args.private_key:
            parser.error("--generate-key cannot be combined with --verify or --private-key")
        private_path, public_path = generate_signing_key(
            args.root.resolve(), args.generate_key, args.public_key
        )
        print(
            json.dumps(
                {
                    "status": "generated",
                    "private_key": str(private_path),
                    "public_key": str(public_path),
                },
                sort_keys=True,
            )
        )
        return
    if args.verify:
        if args.private_key or args.release_policy_amendment:
            parser.error(
                "--private-key/--release-policy-amendment cannot be combined with --verify"
            )
        payload = verify_release(args.root, args.public_key)
    else:
        if args.public_key:
            parser.error("--public-key is only valid with --verify or --generate-key")
        payload = build_release(
            args.root,
            args.private_key,
            release_policy_amendment=args.release_policy_amendment,
        )
    print(json.dumps({"status": payload["status"], "model_sha256": payload["model_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
