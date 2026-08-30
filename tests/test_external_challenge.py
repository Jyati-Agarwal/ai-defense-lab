import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from sim.external_challenge import (
    AGENTDOJO_INTERPRETATION,
    OFFICIAL_AGENTDOJO_EVIDENCE_SHA256,
    OFFICIAL_AGENTDOJO_INTERPRETATION,
    SCORED_FIDELITIES,
    TAXONOMY_SOURCES,
    WATERMARK,
    generate_challenge,
    load_mapping,
    run_challenge,
)
from sim.mandate_twin import (
    SYNTHETIC_IDENTITY_SEED,
    SYNTHETIC_KEY_SCOPE,
    key_registry_sha256,
    observed_event_key_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _challenge_manifest(tmp_path: Path, seed: int) -> Path:
    registry = observed_event_key_registry(generate_challenge(seed=seed))
    path = tmp_path / f"events-{seed}.manifest.json"
    path.write_text(
        json.dumps(
            {
                "synthetic_identity_seed": SYNTHETIC_IDENTITY_SEED,
                "key_registry_scope": SYNTHETIC_KEY_SCOPE,
                "key_registry": registry,
                "key_registry_sha256": key_registry_sha256(registry),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _digest(event: dict) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_external_cases_are_deterministic_schema_valid_and_cover_sources() -> None:
    mapping = load_mapping()
    first = generate_challenge(seed=731)
    second = generate_challenge(seed=731)
    schema = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    assert len(first) == 105
    assert set(mapping["fidelity_definitions"]) == {"exact", "behavioral_proxy", "unsupported"}
    assert Counter(row["fidelity"] for row in mapping["coverage"].values()) == {
        "exact": 12,
        "behavioral_proxy": 19,
        "unsupported": 17,
    }
    assert all(row["outcome_class"] and row["limitation"] for row in mapping["coverage"].values())
    assert all("fraud_classifier" in row for atlas_id, row in mapping["coverage"].items() if atlas_id.startswith("GPF-C"))
    assert [_digest(event) for event in first] == [_digest(event) for event in second]
    assert {event["label"]["atlas_id"] for event in first} == set(mapping["coverage"])
    assert {
        event["external_challenge"]["source_task_id"]
        for event in first
        if event["external_challenge"]["source_family"] == "agentdojo"
    } == {f"injection_task_{index}" for index in range(9)}
    assert sum(event["external_challenge"]["source_family"] == "agentdojo" for event in first) == 9
    for event in first:
        assert not list(validator.iter_errors(event))
        metadata = event["external_challenge"]
        assert metadata["evaluation_only"] is True
        assert metadata["watermark"] == WATERMARK
        assert metadata["raw_payload_included"] is False
        assert metadata["fidelity"] in {"exact", "behavioral_proxy", "unsupported"}
        assert set(metadata["external_taxonomy"]) == set(TAXONOMY_SOURCES)
        assert event["label"]["split"] == "D"
        assert event["label"]["is_fraud"] is True
        if metadata["fidelity"] == "unsupported":
            assert metadata["applied_primitives"] == []

    sources = mapping["sources"]
    assert sources["mitre_atlas"]["commit"] == "2306ecaf04fb7a14be068b97e3ff40c2d9112e28"
    assert sources["mitre_atlas"]["sha256"] == "0e07bb07fc6423d72cdf24ddc2038a6905bcbc00ba571064153119ee1a5888d4"
    assert sources["owasp_agentic"]["sha256"] == "aa2a8563c255ab1f874ccb7db780b5998d20c2c758d92747be008d32fdad668e"
    assert sources["agentdojo"]["commit"] == "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
    assert sources["agentdojo"]["sha256"] == "fa2dedf8e66c95ffadb54f5a26648728b08c56f302858de4ee9a5cc0b3891618"
    assert sources["federal_reserve_fraud_classifier"]["sha256"] == (
        "73907ac5afbcbcf1a92407afcf00d8c5797e26d0056e4688838695ab66873b56"
    )


def test_external_cases_contain_no_operational_payloads_or_secret_like_text() -> None:
    mapping_text = (ROOT / "atlas" / "external_sources.yaml").read_text(encoding="utf-8")
    assert not re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", mapping_text)
    assert not re.search(r"-----BEGIN [A-Z ]+PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}\b", mapping_text)

    visible: list[str] = []
    for event in generate_challenge(seed=991):
        metadata = event["external_challenge"]
        visible.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        chain = event.get("mandate_chain") or {}
        visible.append(str((chain.get("intent") or {}).get("prompt_playback") or ""))
        visible.extend(str(item.get("name") or "") for item in (chain.get("cart") or {}).get("items") or [])
        trace = event.get("agent_trace") or {}
        visible.extend(str(call.get("topic") or "") for call in trace.get("tool_calls") or [])
        for page in trace.get("pages_read") or []:
            sample = str(page.get("hidden_text_sample") or "")
            if sample:
                assert "SAFE SYNTHETIC" in sample
            visible.extend((str(page.get("url") or ""), sample))
    text = "\n".join(visible)
    assert not re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", text)
    assert not re.search(r"-----BEGIN [A-Z ]+PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}\b", text)
    assert not re.search(r"(?i)bearer\s+[A-Za-z0-9._~-]{12,}|password\s*[:=]", text)


def test_runner_excludes_unsupported_and_preserves_frozen_model(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    model = results / "model.pkl"
    model.write_bytes(b"frozen-test-model-v1")
    before_files = {path.name: path.read_bytes() for path in results.iterdir()}
    scored_case_ids: list[str] = []

    def scorer(events: list[dict]) -> list[dict]:
        assert all(event["external_challenge"]["fidelity"] in SCORED_FIDELITIES for event in events)
        scored_case_ids.extend(event["external_challenge"]["case_id"] for event in events)
        actions = ("BLOCK", "HOLD", "STEP_UP", "ALLOW")
        scores = {"BLOCK": 0.91, "HOLD": 0.72, "STEP_UP": 0.45, "ALLOW": 0.01}
        return [
            {
                "fused_score": scores[actions[index % len(actions)]],
                "action": actions[index % len(actions)],
                "reason_codes": ["deterministic mock scorer"],
            }
            for index, _event in enumerate(events)
        ]

    output = tmp_path / "external_attack_challenge.json"
    event_manifest = _challenge_manifest(tmp_path, 1701)
    artifact = run_challenge(
        results,
        output_path=output,
        event_manifest_path=event_manifest,
        seed=1701,
        scorer=scorer,
    )
    rows = artifact["case_results"]
    eligible = [row for row in rows if row["fidelity"] in SCORED_FIDELITIES]
    unsupported = [row for row in rows if row["fidelity"] == "unsupported"]

    assert output.is_file()
    assert artifact["counts"]["total_cases"] == 105
    assert artifact["counts"]["local_atlas_ids"] == 48
    assert artifact["counts"]["agentdojo_outcomes"] == 9
    assert artifact["counts"]["scored_cases"] == len(eligible) == len(scored_case_ids)
    assert artifact["counts"]["unsupported_cases"] == len(unsupported)
    assert artifact["coverage"] == {
        "atlas_vectors": 48,
        "by_fidelity": {"behavioral_proxy": 19, "exact": 12, "unsupported": 17},
        "by_source": {
            "federal_reserve_fraud_classifier": 15,
            "local_attack_atlas": 48,
            "mitre_atlas": 28,
            "owasp_agentic": 28,
        },
        "fidelity_definitions": load_mapping()["fidelity_definitions"],
    }
    assert artifact["recall"]["overall"]["eligible"] == len(eligible)
    assert artifact["recall"]["by_fidelity"]["unsupported"] == {
        "cases": len(unsupported),
        "eligible": 0,
        "detected": 0,
        "detection_recall": None,
        "stopped": 0,
        "stop_recall": None,
    }
    assert artifact["recall"]["overall"]["detected"] > artifact["recall"]["overall"]["stopped"]
    assert artifact["recall"]["overall"]["detection_recall"] > artifact["recall"]["overall"]["stop_recall"]
    assert "STEP_UP" in artifact["recall"]["definitions"]["detection"]
    assert "STEP_UP" not in artifact["recall"]["definitions"]["stop"]
    assert all(row["action"] == "NOT_SCORED" and row["fused_score"] is None for row in unsupported)
    assert len(artifact["misses"]["not_stopped"]) > len(artifact["misses"]["scored_allow"])
    assert all(row["limitation"] for rows in artifact["misses"].values() for row in rows)
    assert artifact["non_contamination"]["training_partitions_read"] == []
    assert artifact["non_contamination"]["training_or_model_selection_writes"] == []
    assert artifact["non_contamination"]["frozen_model_hash_invariant"] is True
    assert artifact["non_contamination"]["model_hash_before"] == artifact["non_contamination"]["model_hash_after"]
    assert artifact["model_hash"] == hashlib.sha256(b"frozen-test-model-v1").hexdigest()
    fixture_manifest = json.loads(event_manifest.read_text(encoding="utf-8"))
    assert artifact["events_manifest_sha256"] == hashlib.sha256(
        event_manifest.read_bytes()
    ).hexdigest()
    assert artifact["key_registry_sha256"] == fixture_manifest["key_registry_sha256"]
    assert {path.name: path.read_bytes() for path in results.iterdir()} == before_files
    assert artifact["sources"]["agentdojo"]["commit"] == "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
    assert artifact["sources"]["federal_reserve_fraud_classifier"]["last_modified"] == "2022-12-29T14:19:36Z"
    assert len(artifact["sources"]["mitre_atlas"]["sha256"]) == 64
    assert {key for row in rows for key in row["source_keys"]} <= set(artifact["sources"])
    assert len(artifact["case_set_sha256"]) == 64
    assert artifact["agentdojo_interpretation"] == AGENTDOJO_INTERPRETATION
    assert "not an AgentDojo task-success or ASR reproduction" in artifact["agentdojo_interpretation"]
    assert "deliberately absent" in artifact["agentdojo_interpretation"]
    official = artifact["official_agentdojo"]
    interpretations = (artifact["agentdojo_interpretation"] + official["interpretation"]).lower()
    assert "genuine official" not in interpretations
    assert "from the pinned official banking run" not in interpretations
    assert "authenticity is pending signed release" in interpretations
    assert official["evidence"] == {
        "file": "results/agentdojo_banking_official.json",
        "artifact_sha256": OFFICIAL_AGENTDOJO_EVIDENCE_SHA256,
        "canonical_sha256_match": True,
        "structural_consistency_valid": True,
        "authenticity_verified": False,
        "authenticity_note": "Require this artifact SHA-256 in the signed release manifest.",
        "raw_logs_verified": False,
        "injection_utility_validation": "aggregate_only",
    }
    assert official["source"]["commit"] == "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
    assert official["source"]["materialized_source_sha256"] == (
        "05cfd031a561bd16794dac706ac6321b0bb887f5d1812880afadc92133a3ac90"
    )
    assert official["model"] == {
        "adapter": "openai-compatible",
        "id": "gpt-4.1-2025-04-14",
        "base_url": "https://api.openai.com/v1",
    }
    assert official["configuration_sha256"] == "000e41d8afd0c7e61b23afd93b97cf0bb2358caa31f0425d69a62533bc893126"
    assert official["configuration"]["benchmark_version"] == "v1.2.2"
    assert official["configuration"]["suite"] == "banking"
    assert official["run"]["attacked_task_pairs"] == 144
    assert official["run"]["official_harness_executed"] is True
    assert official["run"]["official_task_success_evaluators_executed"] is True
    assert official["utility"] == {"successes": 117, "total": 144, "rate": 0.8125}
    assert official["attacker_goal"]["successes"] == 16
    assert official["attacker_goal"]["total"] == 144
    assert official["attacker_goal"]["attack_success_rate"] == 16 / 144
    assert official["interpretation"] == OFFICIAL_AGENTDOJO_INTERPRETATION
    assert "not our translated synthetic-event detector score" in official["interpretation"]


def test_runner_rejects_uncovered_signer_before_custom_scoring(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "model.pkl").write_bytes(b"frozen-test-model-v1")
    event_manifest = _challenge_manifest(tmp_path, 1701)
    payload = json.loads(event_manifest.read_text(encoding="utf-8"))
    payload["key_registry"].pop(next(iter(payload["key_registry"])))
    payload["key_registry_sha256"] = key_registry_sha256(payload["key_registry"])
    event_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    scorer_called = False

    def scorer(_events: list[dict]) -> list[dict]:
        nonlocal scorer_called
        scorer_called = True
        return []

    with pytest.raises(RuntimeError, match="not covered"):
        run_challenge(
            results,
            event_manifest_path=event_manifest,
            seed=1701,
            scorer=scorer,
        )
    assert scorer_called is False


def test_runner_fails_closed_on_unusable_official_evidence(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "model.pkl").write_bytes(b"frozen-test-model-v1")
    source = json.loads((ROOT / "results" / "agentdojo_banking_official.json").read_text(encoding="utf-8"))

    def coherently_tamper(value: dict) -> None:
        value["case_results"][0]["utility"] = not value["case_results"][0]["utility"]
        successes = sum(row["utility"] for row in value["case_results"])
        value["utility"] = {"successes": successes, "total": 144, "rate": successes / 144}

    candidates = [tmp_path / "missing.json"]
    for name, mutate in (
        ("tampered", coherently_tamper),
        ("failed", lambda value: value["fidelity"].update(official_harness_executed=False)),
        ("incomplete", lambda value: value["case_results"].pop()),
    ):
        candidate = tmp_path / f"{name}.json"
        payload = json.loads(json.dumps(source))
        mutate(payload)
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        candidate.chmod(0o444)
        candidates.append(candidate)

    scorer_called = False
    event_manifest = _challenge_manifest(tmp_path, 20260827)

    def scorer(_events: list[dict]) -> list[dict]:
        nonlocal scorer_called
        scorer_called = True
        return []

    for candidate in candidates:
        with pytest.raises(RuntimeError):
            run_challenge(
                results,
                official_evidence_path=candidate,
                event_manifest_path=event_manifest,
                scorer=scorer,
            )
    assert scorer_called is False


def test_runner_fails_if_scorer_mutates_model_and_has_no_training_import(tmp_path: Path) -> None:
    source_path = ROOT / "sim" / "external_challenge.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "defend.train" not in imports
    assert "train_events" not in source
    assert "data/events.jsonl" not in source

    results = tmp_path / "results"
    results.mkdir()
    model = results / "model.pkl"
    model.write_bytes(b"frozen-test-model-v1")
    event_manifest = _challenge_manifest(tmp_path, 44)

    def mutating_scorer(events: list[dict]) -> list[dict]:
        model.write_bytes(b"mutated")
        return [{"fused_score": 0.5, "action": "STEP_UP"} for _event in events]

    with pytest.raises(RuntimeError, match="frozen model changed"):
        run_challenge(
            results,
            event_manifest_path=event_manifest,
            seed=44,
            scorer=mutating_scorer,
        )

    model.write_bytes(b"frozen-test-model-v1")

    def inconsistent_scorer(events: list[dict]) -> list[dict]:
        return [{"fused_score": 0.01, "action": "BLOCK"} for _event in events]

    with pytest.raises(ValueError, match="does not match canonical policy"):
        run_challenge(
            results,
            event_manifest_path=event_manifest,
            seed=44,
            scorer=inconsistent_scorer,
        )
