from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_apprunner.py"
CONFIG = ROOT / "deploy" / "apprunner.json"
DEPLOY_MAKEFILE = ROOT / "deploy" / "Makefile"


def _external_public_key(tmp_path: Path) -> Path:
    path = tmp_path / "release-ed25519.pub.pem"
    path.write_text("public test material\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def test_root_makefile_remains_preregistered() -> None:
    preregistration = json.loads(
        (ROOT / "results" / "evaluation_v4_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    observed = hashlib.sha256((ROOT / "Makefile").read_bytes()).hexdigest()
    assert observed == preregistration["bindings"]["makefile_sha256"]


def test_app_runner_config_is_fail_closed() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["platform"] == "linux/amd64"
    assert payload["container_port"] == 8000
    assert payload["instance"] == {"cpu": "2 vCPU", "memory": "4 GB"}
    assert payload["health_check"] == {
        "protocol": "HTTP",
        "path": "/health",
        "interval_seconds": 10,
        "timeout_seconds": 5,
        "healthy_threshold": 1,
        "unhealthy_threshold": 5,
    }


def test_dry_run_has_no_external_side_effects_or_secret_output(tmp_path: Path) -> None:
    public_key = _external_public_key(tmp_path)
    environment = {
        **os.environ,
        "AI_DEFENSE_RELEASE_PUBLIC_KEY": str(public_key),
        "AWS_REGION": "us-east-1",
        "APP_RUNNER_ECR_ACCESS_ROLE_ARN": "arn:aws:iam::123456789012:role/AppRunnerECRAccess",
        "AWS_SECRET_ACCESS_KEY": "must-not-appear",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT), "--config", str(CONFIG), "--dry-run"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "no AWS, Docker, release, or network command was executed" in result.stdout
    assert "HTTP /health" in result.stdout
    assert "APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED=true" in result.stdout
    assert "must-not-appear" not in result.stdout + result.stderr


def test_missing_aws_credentials_stops_before_docker(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    aws = fake_bin / "aws"
    aws.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >>\"$CALL_LOG\"\nexit 1\n",
        encoding="utf-8",
    )
    aws.chmod(0o755)
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/bin/sh\nprintf 'docker %s\\n' \"$*\" >>\"$CALL_LOG\"\nexit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    public_key = _external_public_key(tmp_path)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CALL_LOG": str(calls),
        "AI_DEFENSE_RELEASE_PUBLIC_KEY": str(public_key),
        "AWS_REGION": "us-east-1",
        "APP_RUNNER_ECR_ACCESS_ROLE_ARN": "arn:aws:iam::123456789012:role/AppRunnerECRAccess",
        "APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED": "true",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT), "--config", str(CONFIG)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "deployment blocked: AWS credentials unavailable; no deployment was attempted"
    )
    observed = calls.read_text(encoding="utf-8")
    assert observed.startswith("sts get-caller-identity")
    assert "docker" not in observed
    assert "ecr" not in observed
    assert "create-service" not in observed


def test_unconfirmed_app_runner_eligibility_stops_before_aws_or_docker(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    for name in ("aws", "docker"):
        executable = fake_bin / name
        executable.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >>\"$CALL_LOG\"\nexit 99\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    public_key = _external_public_key(tmp_path)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CALL_LOG": str(calls),
        "AI_DEFENSE_RELEASE_PUBLIC_KEY": str(public_key),
        "AWS_REGION": "us-east-1",
        "APP_RUNNER_ECR_ACCESS_ROLE_ARN": "arn:aws:iam::123456789012:role/AppRunnerECRAccess",
    }
    environment.pop("APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT), "--config", str(CONFIG)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "existing-customer eligibility is unconfirmed" in result.stderr
    assert not calls.exists()


def test_deployment_make_targets_use_explicit_amendment_and_safe_modes() -> None:
    source = DEPLOY_MAKEFILE.read_text(encoding="utf-8")
    assert "scripts/release.py --release-policy-amendment" in source
    assert "scripts/release.py --verify" in source
    assert "--build-only" in source
    assert "--dry-run" in source
    assert "defend.evaluate" not in source
    assert "sim.population" not in source
