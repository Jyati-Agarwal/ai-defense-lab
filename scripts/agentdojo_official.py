"""Run the pinned official AgentDojo banking benchmark without proxy scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/ethz-spylab/agentdojo.git"
COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
TREE = "45cc40920b8ddca28e675505fdd6a92139008ab3"
BENCHMARK_VERSION = "v1.2.2"
SUITE = "banking"
LICENSE = "MIT"
AGENTDOJO_VERSION = "0.1.35"
LICENSE_SHA256 = "4285a071f2d382338e52b4fb0a186d952984a34d43a33d8872e1a1d8cb43401e"
LOCK_SHA256 = "395e3d0a59214515008d27c4462c9723ab7594374243f561ad5f19b3aa8a5330"
SOURCE_SHA256 = "05cfd031a561bd16794dac706ac6321b0bb887f5d1812880afadc92133a3ac90"
DEFAULT_CHECKOUT = ROOT / "data" / "agentdojo" / f"source-{COMMIT}"
DEFAULT_RUNS = ROOT / "data" / "agentdojo" / "runs"
DEFAULT_OUTPUT = ROOT / "results" / "agentdojo_banking_official.json"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
EXPECTED_USER_TASKS = tuple(f"user_task_{index}" for index in range(16))
EXPECTED_INJECTION_TASKS = tuple(f"injection_task_{index}" for index in range(9))
PINNED_TOP_LEVEL = (".python-version", "LICENSE", "README.md", "pyproject.toml", "uv.lock")
MARKER = "__AGENTDOJO_PREFLIGHT_JSON__="

_OFFICIAL_PREFLIGHT = f"""
import json
from importlib.metadata import version
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite

suite = get_suite({BENCHMARK_VERSION!r}, {SUITE!r})
ground_truth_ok, _ = suite.check(check_injectable=False)
attack = load_attack("injecagent", suite, GroundTruthPipeline(None))
probe_injection = next(iter(suite.injection_tasks.values()))
attack_path_counts = {{task.ID: len(attack.attack(task, probe_injection)) for task in suite.user_tasks.values()}}
key = lambda value: int(value.rsplit('_', 1)[1])
print({MARKER!r} + json.dumps({{
    "agentdojo_version": version("agentdojo"),
    "suite": suite.name,
    "ground_truth_check_passed": ground_truth_ok,
    "attack_path_counts": attack_path_counts,
    "user_tasks": sorted(suite.user_tasks, key=key),
    "injection_tasks": sorted(suite.injection_tasks, key=key),
    "models": sorted(str(value) for value in ModelsEnum),
    "attacks": sorted(ATTACKS),
    "dos_attacks": sorted(name for name, attack in ATTACKS.items() if attack.is_dos_attack),
    "defenses": sorted(DEFENSES),
}}, sort_keys=True))
"""


class AgentDojoError(RuntimeError):
    """A fail-closed source, environment, execution, or evidence failure."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _read_json_object(
    path: Path, *, label: str, max_bytes: int, require_readonly: bool = False
) -> tuple[dict[str, Any], bytes]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or (require_readonly and metadata.st_mode & 0o222)
                or metadata.st_size > max_bytes
            ):
                raise AgentDojoError(f"{label} must be a bounded regular file")
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise AgentDojoError(f"{label} exceeds its size limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
        if type(value) is not dict:
            raise ValueError("root JSON value is not an object")
        return value, raw
    except AgentDojoError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise AgentDojoError(f"{label} is invalid JSON or symbolic") from error


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True, check=False)
    if result.returncode:
        raise AgentDojoError(f"command failed with exit code {result.returncode}: {command[0]}")
    return result


def _git(checkout: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(checkout), *arguments]).stdout.strip()


def _materialized_source_sha256(checkout: Path) -> str:
    tracked = _git(checkout, "ls-files", *PINNED_TOP_LEVEL, "src").splitlines()
    paths = [checkout / name for name in tracked if (checkout / name).is_file()]
    manifest = {path.relative_to(checkout).as_posix(): _sha256_file(path) for path in paths}
    return _sha256_bytes(_canonical(manifest))


def verify_checkout(checkout: Path) -> dict[str, str]:
    """Verify the immutable upstream identity and every materialized runtime input."""
    required = [checkout / name for name in (*PINNED_TOP_LEVEL, "src")]
    if not checkout.is_dir() or not all(path.exists() for path in required):
        raise AgentDojoError(f"incomplete AgentDojo checkout: {checkout}")
    if (checkout / ".env").exists():
        raise AgentDojoError("AgentDojo checkout must not contain an environment file")
    if _git(checkout, "rev-parse", "HEAD") != COMMIT:
        raise AgentDojoError("AgentDojo checkout is not at the pinned commit")
    if _git(checkout, "rev-parse", "HEAD^{tree}") != TREE:
        raise AgentDojoError("AgentDojo checkout tree does not match the pin")
    if _git(checkout, "remote", "get-url", "origin") != REPOSITORY:
        raise AgentDojoError("AgentDojo checkout has an unexpected origin")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise AgentDojoError("AgentDojo checkout contains source changes or untracked files")
    if _sha256_file(checkout / "LICENSE") != LICENSE_SHA256:
        raise AgentDojoError("AgentDojo MIT license does not match the pin")
    if _sha256_file(checkout / "uv.lock") != LOCK_SHA256:
        raise AgentDojoError("AgentDojo lockfile does not match the pin")
    source_sha256 = _materialized_source_sha256(checkout)
    if source_sha256 != SOURCE_SHA256:
        raise AgentDojoError("AgentDojo materialized source does not match the pin")
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "tree": TREE,
        "license": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "uv_lock_sha256": LOCK_SHA256,
        "materialized_source_sha256": source_sha256,
    }


def ensure_checkout(checkout: Path = DEFAULT_CHECKOUT, *, offline: bool = False) -> bool:
    """Return True for a cache hit; otherwise sparsely fetch the exact commit."""
    if checkout.exists():
        verify_checkout(checkout)
        return True
    if offline:
        raise AgentDojoError(f"offline mode requires the pinned checkout cache: {checkout}")
    if shutil.which("git") is None:
        raise AgentDojoError("git is required to fetch the pinned AgentDojo source")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agentdojo-fetch-", dir=checkout.parent) as temporary:
        candidate = Path(temporary) / "source"
        _run(["git", "init", "--quiet", str(candidate)])
        _run(["git", "-C", str(candidate), "remote", "add", "origin", REPOSITORY])
        _run(["git", "-C", str(candidate), "sparse-checkout", "init", "--cone"])
        _run(["git", "-C", str(candidate), "sparse-checkout", "set", "src"])
        _run(
            [
                "git",
                "-C",
                str(candidate),
                "-c",
                "protocol.version=2",
                "fetch",
                "--quiet",
                "--depth=1",
                "--filter=blob:none",
                "origin",
                COMMIT,
            ]
        )
        _run(["git", "-C", str(candidate), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
        verify_checkout(candidate)
        os.replace(candidate, checkout)
    return False


def _env_value(name: str, env_file: Path | None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if env_file is None or not env_file.is_file():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        if key.strip() == name and value.strip().strip("'\""):
            return value.strip().strip("'\"")
    return None


def _uv_command(checkout: Path, *, offline: bool) -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise AgentDojoError("uv is required for AgentDojo's isolated locked environment")
    command = [
        uv,
        "run",
        "--frozen",
        "--no-dev",
        "--isolated",
        "--no-editable",
        "--project",
        str(checkout),
        "--directory",
        str(checkout),
        "--no-env-file",
        "--no-config",
    ]
    if offline:
        command.append("--offline")
    return command


def _controlled_environment() -> dict[str, str]:
    """Keep ambient Python/uv settings from changing the attested execution."""
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("PYTHON", "UV_")) or name in {"VIRTUAL_ENV", "OPENAI_COMPATIBLE_API_KEY"}:
            environment.pop(name)
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def official_inventory(
    checkout: Path = DEFAULT_CHECKOUT,
    *,
    offline: bool = False,
) -> dict[str, Any]:
    source = verify_checkout(checkout)
    result = _run(
        _uv_command(checkout, offline=offline) + ["python", "-c", _OFFICIAL_PREFLIGHT],
        environment=_controlled_environment(),
    )
    verify_checkout(checkout)
    lines = [line[len(MARKER) :] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise AgentDojoError("official AgentDojo preflight did not return one inventory record")
    inventory = json.loads(lines[0])
    if inventory.get("suite") != SUITE or inventory.get("ground_truth_check_passed") is not True:
        raise AgentDojoError("official AgentDojo banking ground-truth evaluator check failed")
    if inventory.get("agentdojo_version") != AGENTDOJO_VERSION:
        raise AgentDojoError("installed AgentDojo package version does not match the source pin")
    if tuple(inventory.get("user_tasks", ())) != EXPECTED_USER_TASKS:
        raise AgentDojoError("official AgentDojo banking user-task inventory changed")
    if tuple(inventory.get("injection_tasks", ())) != EXPECTED_INJECTION_TASKS:
        raise AgentDojoError("official AgentDojo banking injection-task inventory changed")
    attack_path_counts = inventory.get("attack_path_counts", {})
    if set(attack_path_counts) != set(EXPECTED_USER_TASKS) or not all(
        type(count) is int and count > 0 for count in attack_path_counts.values()
    ):
        raise AgentDojoError("official InjecAgent attack path could not inject every banking user task")
    inventory["source"] = source
    return inventory


def _selected(values: Iterable[str], available: tuple[str, ...], label: str) -> tuple[str, ...]:
    requested = tuple(values) or available
    if len(requested) != len(set(requested)):
        raise AgentDojoError(f"duplicate {label} selection")
    unknown = set(requested) - set(available)
    if unknown:
        raise AgentDojoError(f"unknown {label}: {', '.join(sorted(unknown))}")
    return tuple(value for value in available if value in requested)


def build_config(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    if "openai-compatible" not in inventory["models"]:
        raise AgentDojoError("pinned AgentDojo is missing its official OpenAI-compatible adapter")
    if not args.model_id.strip():
        raise AgentDojoError("a concrete OpenAI model id is required")
    if args.attack not in inventory["attacks"] or args.attack == "manual":
        raise AgentDojoError("attack must be a reproducible registered AgentDojo attack")
    if args.attack == "tool_knowledge" or args.attack.startswith("important_instructions"):
        raise AgentDojoError("selected attack requires AgentDojo's legacy named-model registry")
    if args.attack in inventory["dos_attacks"]:
        raise AgentDojoError("DoS attacks do not execute the full injection-task matrix")
    defense = None if args.defense == "none" else args.defense
    if defense is not None and defense not in inventory["defenses"]:
        raise AgentDojoError(f"unknown AgentDojo defense: {defense}")
    if defense in {"tool_filter", "transformers_pi_detector"}:
        raise AgentDojoError(f"{defense} is incompatible with this isolated OpenAI-compatible lane")
    return {
        "protocol": "official_agentdojo_banking_v1",
        "repository_commit": COMMIT,
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE,
        "model_adapter": "openai-compatible",
        "model_id": args.model_id,
        "base_url": DEFAULT_BASE_URL,
        "attack": args.attack,
        "defense": defense,
        "user_tasks": list(_selected(args.user_task, EXPECTED_USER_TASKS, "user task")),
        "injection_tasks": list(_selected(args.injection_task, EXPECTED_INJECTION_TASKS, "injection task")),
    }


def plan(config: dict[str, Any]) -> dict[str, Any]:
    attacked = len(config["user_tasks"]) * len(config["injection_tasks"])
    injection_utility = len(config["injection_tasks"])
    jobs = attacked + injection_utility
    logical_calls_per_pipeline_attempt = 16
    pipeline_attempts_per_job = 3
    tenacity_attempts_per_call = 3
    openai_http_attempts_per_client_call = 3
    logical_calls_per_job = logical_calls_per_pipeline_attempt * pipeline_attempts_per_job
    return {
        "attacked_task_pairs": attacked,
        "injection_task_utility_jobs": injection_utility,
        "total_benchmark_jobs": jobs,
        "logical_completion_call_bounds": {
            "minimum": jobs,
            "maximum": jobs * logical_calls_per_job,
            "basis": (
                "up to three official task-level pipeline attempts; each has one initial completion "
                "plus at most fifteen tool-loop completions"
            ),
        },
        "completion_request_attempt_bounds": {
            "minimum": jobs,
            "maximum": jobs
            * logical_calls_per_job
            * tenacity_attempts_per_call
            * openai_http_attempts_per_client_call,
            "basis": (
                "each completion has up to three AgentDojo Tenacity attempts; the pinned OpenAI client "
                "permits two internal retries plus the initial HTTP attempt"
            ),
        },
        "estimated_cost_usd": None,
        "cost_note": "Provider token usage and price are not recorded by AgentDojo; use the provider usage ledger after execution.",
    }


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    checkout = args.checkout.resolve()
    cache_hit = ensure_checkout(checkout, offline=args.offline)
    inventory = official_inventory(checkout, offline=args.offline)
    config = build_config(args, inventory)
    credential_ready = _env_value("OPENAI_API_KEY", args.env_file) is not None
    report = {
        "ready_for_live_run": credential_ready,
        "credential": {"name": "OPENAI_API_KEY", "present": credential_ready, "value_recorded": False},
        "checkout_cache_hit": cache_hit,
        "dependency_network_policy": "offline" if args.offline else "allowed_for_missing_source_or_packages",
        "isolated_environment": {"manager": "uv", "lock_sha256": LOCK_SHA256, "development_groups": False},
        "agentdojo_version": inventory["agentdojo_version"],
        "source": inventory["source"],
        "config": config,
        "config_sha256": _sha256_bytes(_canonical(config)),
        "plan": plan(config),
        "official_ground_truth_check_passed": True,
        "official_attack_path_check_passed": True,
        "official_inventory": {
            "user_tasks": inventory["user_tasks"],
            "injection_tasks": inventory["injection_tasks"],
        },
    }
    return report, inventory


def _log_digest(records: list[tuple[Path, bytes]], root: Path) -> str:
    digest = hashlib.sha256()
    for path, data in sorted(records):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _is_gitignored(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--no-index", "--", relative],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def aggregate_official_logs(logdir: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected_pairs = {(user, injection) for user in config["user_tasks"] for injection in config["injection_tasks"]}
    expected_utilities = set(config["injection_tasks"])
    attacked: dict[tuple[str, str], dict[str, Any]] = {}
    injection_utilities: dict[str, dict[str, Any]] = {}
    paths = sorted(logdir.rglob("*.json")) if logdir.is_dir() else []
    if logdir.is_symlink() or len(paths) != len(expected_pairs) + len(expected_utilities):
        raise AgentDojoError("official AgentDojo run did not produce exactly the expected task logs")
    records: list[tuple[Path, bytes]] = []
    for path in paths:
        row, raw = _read_json_object(
            path, label=f"official task log {path.name}", max_bytes=1_000_000
        )
        records.append((path, raw))
        if row.get("suite_name") != SUITE or row.get("benchmark_version") != BENCHMARK_VERSION:
            raise AgentDojoError("official task log has an unexpected suite or benchmark version")
        expected_pipeline = config["model_adapter"] + (
            f"-{config['defense']}" if config["defense"] else ""
        )
        if row.get("pipeline_name") != expected_pipeline or row.get("agentdojo_package_version") != AGENTDOJO_VERSION:
            raise AgentDojoError("official task log has an unexpected pipeline or package version")
        if row.get("error") not in (None, ""):
            raise AgentDojoError("official task log contains an execution error")
        if type(row.get("utility")) is not bool or type(row.get("security")) is not bool:
            raise AgentDojoError("official task log is missing boolean evaluator results")
        if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
            raise AgentDojoError("official task log is missing the executed message trace")
        user_task = row.get("user_task_id")
        injection_task = row.get("injection_task_id")
        attack_type = row.get("attack_type")
        if type(user_task) not in {str, type(None)} or type(injection_task) not in {
            str,
            type(None),
        } or type(attack_type) not in {str, type(None)}:
            raise AgentDojoError("official task log has invalid task identifiers")
        if (user_task, injection_task) in expected_pairs and attack_type == config["attack"]:
            key = (user_task, injection_task)
            if key in attacked:
                raise AgentDojoError("duplicate official attacked-task result")
            attacked[key] = row
        elif user_task in expected_utilities and injection_task is None and attack_type is None:
            if user_task in injection_utilities or row["security"] is not True:
                raise AgentDojoError("invalid official injection-task utility result")
            injection_utilities[user_task] = row
        else:
            raise AgentDojoError("unexpected task result in configuration-scoped AgentDojo log directory")
    if set(attacked) != expected_pairs or set(injection_utilities) != expected_utilities:
        raise AgentDojoError("official AgentDojo result matrix is incomplete")

    rows = [
        {
            "user_task_id": user,
            "injection_task_id": injection,
            "utility": attacked[(user, injection)]["utility"],
            "attacker_goal_success": attacked[(user, injection)]["security"],
        }
        for user, injection in sorted(expected_pairs)
    ]
    completion_responses = sum(
        message.get("role") == "assistant"
        for result in (*attacked.values(), *injection_utilities.values())
        for message in result["messages"]
        if isinstance(message, dict)
    )
    return {
        "raw_log_sha256": _log_digest(records, logdir),
        "raw_log_files": len(paths),
        "successful_completion_responses": completion_responses,
        "utility": {
            "successes": sum(row["utility"] for row in rows),
            "total": len(rows),
            "rate": sum(row["utility"] for row in rows) / len(rows),
        },
        "attacker_goal": {
            "successes": sum(row["attacker_goal_success"] for row in rows),
            "total": len(rows),
            "attack_success_rate": sum(row["attacker_goal_success"] for row in rows) / len(rows),
            "official_field": "security",
            "semantics": "True means the official injection-task evaluator found the attacker goal succeeded.",
        },
        "injection_task_utility": {
            "successes": sum(row["utility"] for row in injection_utilities.values()),
            "total": len(injection_utilities),
            "rate": sum(row["utility"] for row in injection_utilities.values()) / len(injection_utilities),
        },
        "case_results": rows,
    }


def verify_evidence(
    path: Path = DEFAULT_OUTPUT, *, raw_logs_root: Path | None = None
) -> dict[str, Any]:
    """Check aggregate consistency; authenticity comes from the signed release."""
    artifact, raw = _read_json_object(
        path,
        label="official AgentDojo evidence",
        max_bytes=2_000_000,
        require_readonly=True,
    )
    expected_keys = {
        "schema_version", "protocol", "generated_at", "run_id", "duration_seconds",
        "source", "configuration", "configuration_sha256", "input_sha256", "fidelity",
        "plan", "raw_log_sha256", "raw_log_files", "successful_completion_responses",
        "utility", "attacker_goal", "injection_task_utility", "case_results", "safety",
    }
    if type(artifact) is not dict or set(artifact) != expected_keys:
        raise AgentDojoError("official AgentDojo evidence schema changed")
    source = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "tree": TREE,
        "license": LICENSE,
        "license_sha256": LICENSE_SHA256,
        "uv_lock_sha256": LOCK_SHA256,
        "materialized_source_sha256": SOURCE_SHA256,
    }
    config = {
        "protocol": "official_agentdojo_banking_v1",
        "repository_commit": COMMIT,
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE,
        "model_adapter": "openai-compatible",
        "model_id": DEFAULT_MODEL,
        "base_url": DEFAULT_BASE_URL,
        "attack": "injecagent",
        "defense": None,
        "user_tasks": list(EXPECTED_USER_TASKS),
        "injection_tasks": list(EXPECTED_INJECTION_TASKS),
    }
    config_sha256 = _sha256_bytes(_canonical(config))
    if (
        artifact["schema_version"] != 1
        or artifact["protocol"] != "official_agentdojo_banking_v1"
        or artifact["source"] != source
        or artifact["configuration"] != config
        or artifact["configuration_sha256"] != config_sha256
        or artifact["input_sha256"] != _sha256_bytes(
            _canonical({"source": source, "configuration": config})
        )
        or artifact["plan"] != plan(config)
        or artifact["fidelity"]
        != {
            "classification": "exact",
            "official_harness_executed": True,
            "official_task_success_evaluators_executed": True,
            "custom_or_proxy_scoring": False,
        }
        or artifact["safety"]
        != {
            "real_payment_systems_contacted": False,
            "official_in_memory_environment": True,
            "raw_prompts_in_artifact": False,
            "raw_logs_gitignored": True,
            "credential_value_recorded": False,
        }
    ):
        raise AgentDojoError("official AgentDojo source, configuration, or safety evidence changed")
    try:
        generated_at = datetime.fromisoformat(artifact["generated_at"])
    except (TypeError, ValueError) as error:
        raise AgentDojoError("official AgentDojo timestamp is invalid") from error
    if (
        generated_at.tzinfo is None
        or type(artifact["run_id"]) is not str
        or re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{16}", artifact["run_id"]) is None
        or type(artifact["duration_seconds"]) not in {int, float}
        or not math.isfinite(artifact["duration_seconds"])
        or artifact["duration_seconds"] <= 0
        or type(artifact["raw_log_sha256"]) is not str
        or len(artifact["raw_log_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in artifact["raw_log_sha256"])
        or artifact["raw_log_files"] != 153
        or type(artifact["successful_completion_responses"]) is not int
        or not 153 <= artifact["successful_completion_responses"] <= 7_344
    ):
        raise AgentDojoError("official AgentDojo execution evidence is invalid")
    expected_pairs = {
        (user_task, injection_task)
        for user_task in EXPECTED_USER_TASKS
        for injection_task in EXPECTED_INJECTION_TASKS
    }
    rows = artifact["case_results"]
    if type(rows) is not list or len(rows) != len(expected_pairs):
        raise AgentDojoError("official AgentDojo case matrix is incomplete")
    observed: dict[tuple[str, str], tuple[bool, bool]] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {
            "user_task_id", "injection_task_id", "utility", "attacker_goal_success"
        }:
            raise AgentDojoError("official AgentDojo case schema changed")
        if type(row["user_task_id"]) is not str or type(row["injection_task_id"]) is not str:
            raise AgentDojoError("official AgentDojo case identifiers are invalid")
        key = (row["user_task_id"], row["injection_task_id"])
        if key not in expected_pairs or key in observed or type(row["utility"]) is not bool or type(
            row["attacker_goal_success"]
        ) is not bool:
            raise AgentDojoError("official AgentDojo case matrix is invalid")
        observed[key] = (row["utility"], row["attacker_goal_success"])
    utility_successes = sum(value[0] for value in observed.values())
    attacker_successes = sum(value[1] for value in observed.values())
    if artifact["utility"] != {
        "successes": utility_successes,
        "total": 144,
        "rate": utility_successes / 144,
    } or artifact["attacker_goal"] != {
        "successes": attacker_successes,
        "total": 144,
        "attack_success_rate": attacker_successes / 144,
        "official_field": "security",
        "semantics": "True means the official injection-task evaluator found the attacker goal succeeded.",
    } or artifact["injection_task_utility"] != {
        "successes": 9,
        "total": 9,
        "rate": 1.0,
    }:
        raise AgentDojoError("official AgentDojo aggregates do not match the exact case matrix")
    raw_logs_verified = False
    if raw_logs_root is not None:
        logdir = raw_logs_root.resolve() / config_sha256 / artifact["run_id"]
        recomputed = aggregate_official_logs(logdir, config)
        for key in (
            "raw_log_sha256",
            "raw_log_files",
            "successful_completion_responses",
            "utility",
            "attacker_goal",
            "injection_task_utility",
            "case_results",
        ):
            if artifact[key] != recomputed[key]:
                raise AgentDojoError("official AgentDojo aggregate differs from its raw task logs")
        raw_logs_verified = True
    return {
        "structural_consistency_valid": True,
        "authenticity_verified": False,
        "authenticity_note": "Require this artifact SHA-256 in the signed release manifest.",
        "raw_logs_verified": raw_logs_verified,
        "injection_utility_validation": (
            "exact_raw_logs" if raw_logs_verified else "aggregate_only"
        ),
        "artifact_sha256": _sha256_bytes(raw),
        "configuration_sha256": config_sha256,
        "attacked_task_pairs": len(observed),
        "utility": artifact["utility"],
        "attacker_goal": artifact["attacker_goal"],
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise AgentDojoError(f"refusing to overwrite existing evidence artifact: {output}")
    report, _ = preflight(args)
    if not report["ready_for_live_run"]:
        raise AgentDojoError("OPENAI_API_KEY is absent; no model calls were attempted")
    config = report["config"]
    config_sha256 = report["config_sha256"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + secrets.token_hex(8)
    logdir = args.runs.resolve() / config_sha256 / run_id
    logdir.mkdir(parents=True, exist_ok=False)
    command = _uv_command(args.checkout.resolve(), offline=args.offline) + [
        "python",
        "-m",
        "agentdojo.scripts.benchmark",
        "--suite",
        SUITE,
        "--benchmark-version",
        BENCHMARK_VERSION,
        "--model",
        config["model_adapter"],
        "--model-id",
        config["model_id"],
        "--attack",
        config["attack"],
        "--logdir",
        str(logdir),
        "--max-workers",
        "1",
    ]
    if config["defense"] is not None:
        command.extend(["--defense", config["defense"]])
    for task in config["user_tasks"]:
        command.extend(["--user-task", task])
    for task in config["injection_tasks"]:
        command.extend(["--injection-task", task])
    started_at = datetime.now(timezone.utc)
    environment = _controlled_environment()
    environment["OPENAI_COMPATIBLE_API_KEY"] = _env_value("OPENAI_API_KEY", args.env_file) or ""
    environment["OPENAI_COMPATIBLE_BASE_URL"] = config["base_url"]
    _run(command, environment=environment)
    evidence = aggregate_official_logs(logdir, config)
    source = verify_checkout(args.checkout.resolve())
    artifact = {
        "schema_version": 1,
        "protocol": "official_agentdojo_banking_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "duration_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
        "source": source,
        "configuration": config,
        "configuration_sha256": config_sha256,
        "input_sha256": _sha256_bytes(_canonical({"source": source, "configuration": config})),
        "fidelity": {
            "classification": "exact",
            "official_harness_executed": True,
            "official_task_success_evaluators_executed": True,
            "custom_or_proxy_scoring": False,
        },
        "plan": report["plan"],
        **evidence,
        "safety": {
            "real_payment_systems_contacted": False,
            "official_in_memory_environment": True,
            "raw_prompts_in_artifact": False,
            "raw_logs_gitignored": _is_gitignored(logdir),
            "credential_value_recorded": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.chmod(0o444)
    os.replace(temporary, output)
    return artifact


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="disable source/package network fetches; a live run still contacts the fixed OpenAI API",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--attack", default="injecagent")
    parser.add_argument("--defense", default="none")
    parser.add_argument("--user-task", action="append", default=[])
    parser.add_argument("--injection-task", action="append", default=[])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="fetch, lock-check, and inspect the official suite")
    _add_common_arguments(preflight_parser)
    run_parser = commands.add_parser("run", help="execute and aggregate the official benchmark")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = commands.add_parser("verify", help="validate aggregate consistency without model calls")
    verify_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument(
        "--raw-logs",
        action="store_true",
        help="also recompute the exact matrix, including all nine utility jobs, from local raw logs",
    )
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            report, _ = preflight(args)
            print(json.dumps(report, indent=2, sort_keys=True))
            if not report["ready_for_live_run"]:
                raise SystemExit(2)
        elif args.command == "run":
            artifact = run_benchmark(args)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "utility": artifact["utility"],
                        "attacker_goal": artifact["attacker_goal"],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    verify_evidence(
                        args.output,
                        raw_logs_root=DEFAULT_RUNS if args.raw_logs else None,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
    except AgentDojoError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
