#!/usr/bin/env python3
"""Build, verify, and deploy the signed release to AWS App Runner.

The deployment path consumes the already signed amended release. It never runs
simulation, training, arena, vaccination, or evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deploy" / "apprunner.json"
CONTAINER_PUBLIC_KEY = Path("/run/ai-defense/release-ed25519.pub.pem")
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-_]{3,39}\Z")
REPOSITORY_RE = re.compile(
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*[a-z0-9]+(?:[._-][a-z0-9]+)*\Z"
)
REGION_RE = re.compile(r"[a-z]{2}(?:-[a-z]+)+-[0-9]\Z")
ROLE_RE = re.compile(r"arn:aws:iam::([0-9]{12}):role/(.+)\Z")
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DeploymentBlocked(RuntimeError):
    """A precondition failed before a trustworthy deployment completed."""


@dataclass(frozen=True)
class DeployConfig:
    service_name: str
    ecr_repository: str
    local_image: str
    platform: str
    container_port: int
    cpu: str
    memory: str
    health: dict[str, Any]
    operation_attempts: int
    operation_interval_seconds: int
    local_health_attempts: int
    public_health_attempts: int
    health_interval_seconds: int


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentBlocked(f"{label} must be a JSON object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeploymentBlocked(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise DeploymentBlocked(f"{label} must be between {minimum} and {maximum}")
    return value


def load_config(path: Path) -> DeployConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentBlocked(f"deployment config is unreadable: {path}") from error
    source = _object(payload, "deployment config")
    expected = {
        "schema_version",
        "service_name",
        "ecr_repository",
        "local_image",
        "platform",
        "container_port",
        "instance",
        "health_check",
        "wait",
    }
    if set(source) != expected or source.get("schema_version") != 1:
        raise DeploymentBlocked("deployment config schema is unsupported")

    service_name = _string(source["service_name"], "service_name")
    repository = _string(source["ecr_repository"], "ecr_repository")
    local_image = _string(source["local_image"], "local_image")
    platform = _string(source["platform"], "platform")
    if not SERVICE_RE.fullmatch(service_name):
        raise DeploymentBlocked("service_name is not valid for App Runner")
    if not REPOSITORY_RE.fullmatch(repository):
        raise DeploymentBlocked("ecr_repository is not valid for ECR")
    if platform != "linux/amd64":
        raise DeploymentBlocked("platform must be linux/amd64")

    instance = _object(source["instance"], "instance")
    if set(instance) != {"cpu", "memory"}:
        raise DeploymentBlocked("instance has unexpected fields")
    cpu = _string(instance["cpu"], "instance.cpu")
    memory = _string(instance["memory"], "instance.memory")
    supported_instance_pairs = {
        ("0.25 vCPU", "0.5 GB"),
        ("0.25 vCPU", "1 GB"),
        ("0.5 vCPU", "1 GB"),
        ("1 vCPU", "2 GB"),
        ("1 vCPU", "3 GB"),
        ("1 vCPU", "4 GB"),
        ("2 vCPU", "4 GB"),
        ("2 vCPU", "6 GB"),
        ("4 vCPU", "8 GB"),
        ("4 vCPU", "10 GB"),
        ("4 vCPU", "12 GB"),
    }
    if (cpu, memory) not in supported_instance_pairs:
        raise DeploymentBlocked("instance CPU/memory pair is not supported by App Runner")

    health_source = _object(source["health_check"], "health_check")
    if set(health_source) != {
        "protocol",
        "path",
        "interval_seconds",
        "timeout_seconds",
        "healthy_threshold",
        "unhealthy_threshold",
    }:
        raise DeploymentBlocked("health_check has unexpected fields")
    path_value = _string(health_source["path"], "health_check.path")
    if health_source.get("protocol") != "HTTP" or path_value != "/health":
        raise DeploymentBlocked("App Runner must use HTTP /health checks")
    health = {
        "Protocol": "HTTP",
        "Path": path_value,
        "Interval": _integer(health_source["interval_seconds"], "health interval", 1, 20),
        "Timeout": _integer(health_source["timeout_seconds"], "health timeout", 1, 20),
        "HealthyThreshold": _integer(
            health_source["healthy_threshold"], "healthy threshold", 1, 20
        ),
        "UnhealthyThreshold": _integer(
            health_source["unhealthy_threshold"], "unhealthy threshold", 1, 20
        ),
    }
    if health["Timeout"] >= health["Interval"]:
        raise DeploymentBlocked("health timeout must be shorter than its interval")

    wait = _object(source["wait"], "wait")
    if set(wait) != {
        "operation_attempts",
        "operation_interval_seconds",
        "local_health_attempts",
        "public_health_attempts",
        "health_interval_seconds",
    }:
        raise DeploymentBlocked("wait has unexpected fields")
    return DeployConfig(
        service_name=service_name,
        ecr_repository=repository,
        local_image=local_image,
        platform=platform,
        container_port=_integer(source["container_port"], "container_port", 1, 65535),
        cpu=cpu,
        memory=memory,
        health=health,
        operation_attempts=_integer(wait["operation_attempts"], "operation attempts", 1, 360),
        operation_interval_seconds=_integer(
            wait["operation_interval_seconds"], "operation interval", 1, 60
        ),
        local_health_attempts=_integer(
            wait["local_health_attempts"], "local health attempts", 1, 600
        ),
        public_health_attempts=_integer(
            wait["public_health_attempts"], "public health attempts", 1, 120
        ),
        health_interval_seconds=_integer(wait["health_interval_seconds"], "health interval", 1, 60),
    )


def _external_public_key(root: Path) -> Path:
    configured = os.environ.get("AI_DEFENSE_RELEASE_PUBLIC_KEY")
    if not configured:
        raise DeploymentBlocked("AI_DEFENSE_RELEASE_PUBLIC_KEY is not configured")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise DeploymentBlocked("release public key must be an absolute, non-symlink path")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise DeploymentBlocked("release public key is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise DeploymentBlocked("release public key must be outside the repository")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise DeploymentBlocked("release public key must be a non-writable regular file")
    return resolved


def _required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise DeploymentBlocked(f"required command is missing: {name}")
    return path


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    quiet: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL if quiet else None,
        stderr=subprocess.PIPE if capture or quiet else None,
    )
    if check and result.returncode != 0:
        label = " ".join(command[:2])
        raise DeploymentBlocked(f"command failed: {label}")
    return result


def _aws(
    aws: str,
    region: str,
    arguments: Sequence[str],
    *,
    query: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [aws, *arguments, "--region", region, "--no-cli-pager"]
    if query is not None:
        command.extend(["--query", query, "--output", "text"])
    else:
        command.extend(["--output", "json"])
    return _run(command, capture=True, check=check)


def _host_release_preflight(root: Path, python: Path, public_key: Path) -> str:
    preregistration = root / "results" / "evaluation_v4_preregistration.json"
    try:
        prereg = json.loads(preregistration.read_text(encoding="utf-8"))
        expected_makefile = prereg["bindings"]["makefile_sha256"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DeploymentBlocked("V4 preregistration has no Makefile custody binding") from error
    makefile_hash = hashlib.sha256((root / "Makefile").read_bytes()).hexdigest()
    if makefile_hash != expected_makefile:
        raise DeploymentBlocked("root Makefile no longer matches the V4 preregistration")
    _run(
        [
            str(python),
            str(root / "scripts" / "release.py"),
            "--root",
            str(root),
            "--verify",
            "--public-key",
            str(public_key),
        ],
        quiet=True,
    )
    return hashlib.sha256((root / "results" / "release_manifest.json").read_bytes()).hexdigest()


def _docker_restrictions() -> list[str]:
    return [
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]


def _verify_packaged_image(docker: str, image: str, config: DeployConfig) -> None:
    verification = (
        "test ! -e /app/data/events.jsonl; "
        "test ! -e /app/sim/attacks/heldout; "
        "exec python /app/scripts/release.py --root /app --verify "
        f"--public-key {CONTAINER_PUBLIC_KEY}"
    )
    _run(
        [docker, "run", "--rm", *_docker_restrictions(), image, "sh", "-eu", "-c", verification],
        quiet=True,
    )

    container = f"ai-defense-apprunner-preflight-{os.getpid()}"
    _run(
        [
            docker,
            "run",
            "--detach",
            "--name",
            container,
            *_docker_restrictions(),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            image,
        ],
        quiet=True,
    )
    try:
        probe = (
            "import json,urllib.request; "
            "assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',"
            "timeout=3))['status']=='ready'"
        )
        for _ in range(config.local_health_attempts):
            response = _run(
                [docker, "exec", container, "python", "-c", probe],
                quiet=True,
                check=False,
            )
            if response.returncode == 0:
                return
            running = _run(
                [docker, "inspect", "--format", "{{.State.Running}}", container],
                capture=True,
                check=False,
            )
            if running.returncode != 0 or running.stdout.strip() != "true":
                raise DeploymentBlocked("packaged image exited before /health became ready")
            time.sleep(1)
        raise DeploymentBlocked("packaged image did not make /health ready")
    finally:
        _run([docker, "rm", "--force", container], quiet=True, check=False)


def _build_packaged_image(
    root: Path,
    docker: str,
    public_key: Path,
    release_hash: str,
    config: DeployConfig,
    image_override: str | None,
) -> tuple[str, str]:
    image = image_override or config.local_image
    suffix = release_hash[:20]
    base_image = f"ai-defense-lab-build:{suffix}"
    _run(
        [docker, "build", "--platform", config.platform, "--tag", base_image, str(root)]
    )
    with tempfile.TemporaryDirectory(prefix="ai-defense-apprunner-") as temporary:
        context = Path(temporary)
        key_copy = context / "release-ed25519.pub.pem"
        shutil.copyfile(public_key, key_copy)
        key_copy.chmod(0o444)
        (context / "Dockerfile").write_text(
            "ARG BASE_IMAGE\n"
            "FROM ${BASE_IMAGE}\n"
            "USER root\n"
            "COPY --chown=0:0 --chmod=0444 release-ed25519.pub.pem "
            f"{CONTAINER_PUBLIC_KEY}\n"
            "USER appuser\n"
            f"ENV AI_DEFENSE_RELEASE_PUBLIC_KEY={CONTAINER_PUBLIC_KEY}\n",
            encoding="utf-8",
        )
        _run(
            [
                docker,
                "build",
                "--platform",
                config.platform,
                "--build-arg",
                f"BASE_IMAGE={base_image}",
                "--tag",
                image,
                str(context),
            ]
        )
    _verify_packaged_image(docker, image, config)
    image_id = _run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image], capture=True
    ).stdout.strip()
    if not SHA256_RE.fullmatch(image_id):
        raise DeploymentBlocked("Docker returned an invalid packaged-image ID")
    return image, f"release-{release_hash[:12]}-{image_id.removeprefix('sha256:')[:12]}"


def _credentials_preflight(aws: str, region: str, role_arn: str) -> str:
    identity = _aws(
        aws,
        region,
        ["sts", "get-caller-identity"],
        query="Account",
        check=False,
    )
    account = identity.stdout.strip() if identity.returncode == 0 else ""
    if not re.fullmatch(r"[0-9]{12}", account):
        raise DeploymentBlocked("AWS credentials unavailable; no deployment was attempted")
    match = ROLE_RE.fullmatch(role_arn)
    if match is None or match.group(1) != account:
        raise DeploymentBlocked("App Runner ECR access role must belong to the active AWS account")
    role_name = role_arn.rsplit("/", 1)[-1]
    role = _aws(
        aws,
        region,
        ["iam", "get-role", "--role-name", role_name],
        check=False,
    )
    policies = _aws(
        aws,
        region,
        ["iam", "list-attached-role-policies", "--role-name", role_name],
        check=False,
    )
    if role.returncode != 0 or policies.returncode != 0:
        raise DeploymentBlocked("App Runner ECR access role could not be verified")
    try:
        role_payload = json.loads(role.stdout)["Role"]
        policy_payload = json.loads(policies.stdout)["AttachedPolicies"]
        statements = role_payload["AssumeRolePolicyDocument"]["Statement"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DeploymentBlocked("App Runner ECR access role metadata is invalid") from error
    if isinstance(statements, Mapping):
        statements = [statements]

    def values(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return set(value)
        return set()

    trusted = False
    if isinstance(statements, list):
        for statement in statements:
            if not isinstance(statement, Mapping):
                continue
            principal = statement.get("Principal")
            services = principal.get("Service") if isinstance(principal, Mapping) else None
            if (
                statement.get("Effect") == "Allow"
                and "sts:AssumeRole" in values(statement.get("Action"))
                and "build.apprunner.amazonaws.com" in values(services)
            ):
                trusted = True
                break
    managed_policy = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
    attached = {
        item.get("PolicyArn")
        for item in policy_payload
        if isinstance(item, Mapping)
    }
    if role_payload.get("Arn") != role_arn or not trusted or managed_policy not in attached:
        raise DeploymentBlocked("App Runner ECR access role trust or managed policy is invalid")
    entitlement = _aws(
        aws,
        region,
        ["apprunner", "list-services", "--max-results", "1"],
        check=False,
    )
    if entitlement.returncode != 0:
        raise DeploymentBlocked(
            "App Runner is unavailable to this account/region; no deployment was attempted"
        )
    return account


def _existing_customer_preflight() -> None:
    """Require an explicit eligibility assertion before any AWS API call.

    App Runner has no read-only API that proves whether an account was already
    an App Runner customer at the service's March 31, 2026 cutoff.  Requiring
    this assertion keeps a new or unreviewed account from reaching mutation.
    """
    if os.environ.get("APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED") != "true":
        raise DeploymentBlocked(
            "App Runner existing-customer eligibility is unconfirmed; set "
            "APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED=true only after account review"
        )


def _ecr_repository(aws: str, region: str, repository: str) -> str:
    described = _aws(
        aws,
        region,
        ["ecr", "describe-repositories", "--repository-names", repository],
        query="repositories[0].repositoryUri",
        check=False,
    )
    if described.returncode == 0 and described.stdout.strip() not in {"", "None"}:
        return described.stdout.strip()
    if "RepositoryNotFoundException" not in described.stderr:
        raise DeploymentBlocked("ECR repository lookup failed")
    created = _aws(
        aws,
        region,
        [
            "ecr",
            "create-repository",
            "--repository-name",
            repository,
            "--image-tag-mutability",
            "IMMUTABLE",
            "--image-scanning-configuration",
            "scanOnPush=true",
            "--encryption-configuration",
            "encryptionType=AES256",
        ],
        query="repository.repositoryUri",
    )
    return created.stdout.strip()


def _ecr_login(aws: str, docker: str, region: str, registry: str) -> None:
    aws_process = subprocess.Popen(
        [aws, "ecr", "get-login-password", "--region", region, "--no-cli-pager"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert aws_process.stdout is not None
    docker_process = subprocess.run(
        [docker, "login", "--username", "AWS", "--password-stdin", registry],
        stdin=aws_process.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    aws_process.stdout.close()
    assert aws_process.stderr is not None
    aws_process.stderr.read()
    aws_returncode = aws_process.wait()
    if aws_returncode != 0 or docker_process.returncode != 0:
        raise DeploymentBlocked("ECR authentication failed")


def _push_image(
    aws: str,
    docker: str,
    region: str,
    repository_uri: str,
    local_image: str,
    image_tag: str,
) -> str:
    if not re.fullmatch(r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/.+", repository_uri):
        raise DeploymentBlocked("ECR returned an unexpected repository URI")
    registry = repository_uri.split("/", 1)[0]
    remote = f"{repository_uri}:{image_tag}"
    _ecr_login(aws, docker, region, registry)
    _run([docker, "tag", local_image, remote])
    _run([docker, "push", remote])
    digest = _aws(
        aws,
        region,
        [
            "ecr",
            "describe-images",
            "--repository-name",
            repository_uri.split("/", 1)[1],
            "--image-ids",
            f"imageTag={image_tag}",
        ],
        query="imageDetails[0].imageDigest",
    ).stdout.strip()
    if not SHA256_RE.fullmatch(digest):
        raise DeploymentBlocked("ECR returned an invalid image digest")
    return f"{repository_uri}@{digest}"


def _source_configuration(image_identifier: str, role_arn: str, port: int) -> dict[str, Any]:
    return {
        "AuthenticationConfiguration": {"AccessRoleArn": role_arn},
        "AutoDeploymentsEnabled": False,
        "ImageRepository": {
            "ImageIdentifier": image_identifier,
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {"Port": str(port)},
        },
    }


def _service_summary(aws: str, region: str, service_name: str) -> dict[str, Any] | None:
    response = _aws(aws, region, ["apprunner", "list-services"])
    try:
        summaries = json.loads(response.stdout)["ServiceSummaryList"]
        matches = [item for item in summaries if item.get("ServiceName") == service_name]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DeploymentBlocked("App Runner returned an invalid service list") from error
    if len(matches) > 1:
        raise DeploymentBlocked("App Runner returned duplicate service names")
    return matches[0] if matches else None


def _deploy_service(
    aws: str,
    region: str,
    role_arn: str,
    image_identifier: str,
    release_hash: str,
    config: DeployConfig,
) -> tuple[str, str]:
    source = _source_configuration(image_identifier, role_arn, config.container_port)
    common = {
        "SourceConfiguration": source,
        "InstanceConfiguration": {"Cpu": config.cpu, "Memory": config.memory},
        "HealthCheckConfiguration": config.health,
        "NetworkConfiguration": {
            "EgressConfiguration": {"EgressType": "DEFAULT"},
            "IngressConfiguration": {"IsPubliclyAccessible": True},
            "IpAddressType": "IPV4",
        },
    }
    existing = _service_summary(aws, region, config.service_name)
    if existing is None:
        request = {
            "ServiceName": config.service_name,
            **common,
            "Tags": [{"Key": "release-manifest-sha256", "Value": release_hash}],
        }
        action = "create-service"
    else:
        service_arn = _string(existing.get("ServiceArn"), "existing service ARN")
        status = _string(existing.get("Status"), "existing service status")
        if status != "RUNNING":
            raise DeploymentBlocked(f"existing App Runner service is not RUNNING: {status}")
        request = {"ServiceArn": service_arn, **common}
        action = "update-service"

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as handle:
        json.dump(request, handle, separators=(",", ":"), sort_keys=True)
        handle.flush()
        response = _aws(
            aws,
            region,
            ["apprunner", action, "--cli-input-json", f"file://{handle.name}"],
        )
    try:
        payload = json.loads(response.stdout)
        operation_id = payload["OperationId"]
        service_arn = payload["Service"]["ServiceArn"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DeploymentBlocked("App Runner returned an invalid deployment response") from error
    return _string(service_arn, "service ARN"), _string(operation_id, "operation ID")


def _wait_for_operation(
    aws: str,
    region: str,
    service_arn: str,
    operation_id: str,
    config: DeployConfig,
) -> None:
    terminal = {"SUCCEEDED", "FAILED", "ROLLBACK_SUCCEEDED", "ROLLBACK_FAILED"}
    for _ in range(config.operation_attempts):
        response = _aws(
            aws,
            region,
            ["apprunner", "list-operations", "--service-arn", service_arn],
        )
        try:
            operations = json.loads(response.stdout)["OperationSummaryList"]
            status = next(item["Status"] for item in operations if item["Id"] == operation_id)
        except StopIteration:
            status = None
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise DeploymentBlocked("App Runner returned an invalid operation list") from error
        if status == "SUCCEEDED":
            return
        if status in terminal:
            raise DeploymentBlocked(f"App Runner deployment operation ended with {status}")
        time.sleep(config.operation_interval_seconds)
    raise DeploymentBlocked("App Runner deployment operation timed out")


def _deployed_service(
    aws: str,
    region: str,
    service_arn: str,
    image_identifier: str,
    config: DeployConfig,
) -> str:
    response = _aws(
        aws,
        region,
        ["apprunner", "describe-service", "--service-arn", service_arn],
    )
    try:
        service = json.loads(response.stdout)["Service"]
        observed_image = service["SourceConfiguration"]["ImageRepository"]["ImageIdentifier"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DeploymentBlocked("App Runner returned an invalid service description") from error
    if (
        service.get("Status") != "RUNNING"
        or observed_image != image_identifier
        or service.get("HealthCheckConfiguration") != config.health
    ):
        raise DeploymentBlocked("App Runner service does not match the requested release")
    service_url = _string(service.get("ServiceUrl"), "service URL")
    if not re.fullmatch(r"[A-Za-z0-9.-]+\.awsapprunner\.com", service_url):
        raise DeploymentBlocked("App Runner returned an invalid public URL")
    return f"https://{service_url}"


def _public_smoke(url: str, config: DeployConfig) -> None:
    for _ in range(config.public_health_attempts):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
                payload = json.load(response)
            if response.status == 200 and payload == {"status": "ready"}:
                break
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(config.health_interval_seconds)
    else:
        raise DeploymentBlocked("public App Runner /health smoke did not become ready")
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise DeploymentBlocked("public App Runner root smoke failed")
    except urllib.error.URLError as error:
        raise DeploymentBlocked("public App Runner root smoke failed") from error


def _plan(config: DeployConfig, region: str) -> None:
    print("dry run only; no AWS, Docker, release, or network command was executed")
    print(f"1. verify the signed amended release and frozen root Makefile ({region})")
    print("2. build a clean image, inject only the external public key, and verify it offline")
    print(f"3. push a release-specific image to ECR repository {config.ecr_repository}")
    print(f"4. create or update App Runner service {config.service_name} with HTTP /health")
    print(
        "5. wait for the exact operation, verify image digest/configuration, "
        "and smoke the public URL"
    )
    print("deployment additionally requires APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED=true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", help="override the local packaged-image tag")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the non-mutating plan",
    )
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="build and locally verify; do not call AWS",
    )
    args = parser.parse_args(argv)

    try:
        root = args.root.resolve(strict=True)
        config = load_config(args.config.resolve(strict=True))
        public_key = _external_public_key(root)
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
        role_arn = os.environ.get("APP_RUNNER_ECR_ACCESS_ROLE_ARN", "")
        if args.dry_run:
            if not REGION_RE.fullmatch(region):
                raise DeploymentBlocked("AWS_REGION is missing or invalid")
            if ROLE_RE.fullmatch(role_arn) is None:
                raise DeploymentBlocked("APP_RUNNER_ECR_ACCESS_ROLE_ARN is missing or invalid")
            _plan(config, region)
            return 0

        python = root / ".venv" / "bin" / "python"
        if not python.is_file() or not os.access(python, os.X_OK):
            raise DeploymentBlocked("project Python runtime is missing; run make setup")
        docker = _required_tool("docker")
        aws = ""
        if not args.build_only:
            if not REGION_RE.fullmatch(region):
                raise DeploymentBlocked("AWS_REGION is missing or invalid")
            if ROLE_RE.fullmatch(role_arn) is None:
                raise DeploymentBlocked("APP_RUNNER_ECR_ACCESS_ROLE_ARN is missing or invalid")
            _existing_customer_preflight()
            aws = _required_tool("aws")
            _credentials_preflight(aws, region, role_arn)

        release_hash = _host_release_preflight(root, python, public_key)
        local_image, image_tag = _build_packaged_image(
            root, docker, public_key, release_hash, config, args.image
        )
        if args.build_only:
            print(f"packaged release verified: {local_image}")
            return 0

        repository_uri = _ecr_repository(aws, region, config.ecr_repository)
        image_identifier = _push_image(
            aws, docker, region, repository_uri, local_image, image_tag
        )
        service_arn, operation_id = _deploy_service(
            aws, region, role_arn, image_identifier, release_hash, config
        )
        _wait_for_operation(aws, region, service_arn, operation_id, config)
        public_url = _deployed_service(
            aws, region, service_arn, image_identifier, config
        )
        _public_smoke(public_url, config)
        print(f"deployment ready: {public_url}")
        return 0
    except (DeploymentBlocked, OSError) as error:
        print(f"deployment blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
