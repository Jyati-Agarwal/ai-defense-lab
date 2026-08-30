import os
from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.agentdojo_official import (
    BENCHMARK_VERSION,
    COMMIT,
    DEFAULT_CHECKOUT,
    EXPECTED_INJECTION_TASKS,
    EXPECTED_USER_TASKS,
    LICENSE,
    SOURCE_SHA256,
    TREE,
    plan,
    preflight,
    verify_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_agentdojo_pin_and_scope_are_explicit() -> None:
    assert len(COMMIT) == len(TREE) == 40
    assert len(SOURCE_SHA256) == 64
    assert BENCHMARK_VERSION == "v1.2.2"
    assert LICENSE == "MIT"
    assert EXPECTED_USER_TASKS == tuple(f"user_task_{index}" for index in range(16))
    assert EXPECTED_INJECTION_TASKS == tuple(f"injection_task_{index}" for index in range(9))

    full_plan = plan(
        {
            "user_tasks": list(EXPECTED_USER_TASKS),
            "injection_tasks": list(EXPECTED_INJECTION_TASKS),
        }
    )
    assert full_plan["total_benchmark_jobs"] == 153
    assert full_plan["logical_completion_call_bounds"]["minimum"] == 153
    assert full_plan["logical_completion_call_bounds"]["maximum"] == 7_344
    assert full_plan["completion_request_attempt_bounds"]["maximum"] == 66_096


def test_official_agentdojo_frozen_evidence_and_tamper(tmp_path: Path) -> None:
    source = ROOT / "results" / "agentdojo_banking_official.json"
    verified = verify_evidence(source)
    assert verified["structural_consistency_valid"] is True
    assert verified["authenticity_verified"] is False
    assert verified["injection_utility_validation"] == "aggregate_only"
    assert verified["attacked_task_pairs"] == 144

    artifact = json.loads(source.read_text(encoding="utf-8"))
    artifact["utility"]["successes"] += 1
    tampered = tmp_path / "agentdojo.json"
    tampered.write_text(json.dumps(artifact), encoding="utf-8")
    tampered.chmod(0o444)
    with pytest.raises(RuntimeError, match="aggregates"):
        verify_evidence(tampered)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        source.read_text(encoding="utf-8").replace(
            '  "schema_version": 1,',
            '  "schema_version": 1,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )
    duplicate.chmod(0o444)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        verify_evidence(duplicate)


@pytest.mark.skipif(os.environ.get("AGENTDOJO_LIVE_TEST") != "1", reason="explicit pinned-upstream integration test")
def test_official_agentdojo_live_preflight_from_locked_cache() -> None:
    report, _ = preflight(
        Namespace(
            checkout=DEFAULT_CHECKOUT,
            runs=ROOT / "data" / "agentdojo" / "runs",
            env_file=ROOT / ".env",
            offline=True,
            model_id="gpt-4.1-2025-04-14",
            attack="injecagent",
            defense="none",
            user_task=["user_task_0"],
            injection_task=["injection_task_0"],
        )
    )

    assert report["ready_for_live_run"] is True
    assert report["official_ground_truth_check_passed"] is True
    assert report["official_attack_path_check_passed"] is True
    assert report["source"]["commit"] == COMMIT
    assert report["source"]["materialized_source_sha256"] == SOURCE_SHA256
    assert report["plan"]["attacked_task_pairs"] == 1
    assert report["plan"]["total_benchmark_jobs"] == 2
    assert report["credential"]["value_recorded"] is False
