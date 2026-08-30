import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from sim.fidelity import run_fidelity
from sim.mandate_twin import (
    SYNTHETIC_IDENTITY_SEED,
    key_registry_sha256,
    public_key_fingerprint,
    verify_chain,
)
from sim.population import (
    COPULA_PRIMITIVES,
    COPULA_QUANTILE_PROBABILITIES,
    SINGLETON_PRIMITIVES,
    BehaviorProfile,
    PopulationSimulator,
    generate_population_dataset,
)
import sim.population as population_module
from tests.test_release import _evaluation_preregistration


ROOT = Path(__file__).resolve().parents[1]
EVENT_SCHEMA = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))


def _observed_key_registry(events: list[dict]) -> dict[str, str]:
    registry = {}
    for event in events:
        chain = event.get("mandate_chain")
        if not chain:
            continue
        for principal, part in (
            (f"user:{chain['intent']['user_id']}", chain["intent"]),
            (f"merchant:{chain['cart']['merchant_id']}", chain["cart"]),
            (f"agent:{chain['payment']['agent_id']}", chain["payment"]),
        ):
            registry[principal] = public_key_fingerprint(part["signer_public_key"])
    return dict(sorted(registry.items()))


def test_seeded_population_is_stateful_and_contract_valid() -> None:
    first = PopulationSimulator(seed=42, customers=30, merchants_per_category=1, agentic_rate=0)
    second = PopulationSimulator(seed=42, customers=30, merchants_per_category=1, agentic_rate=0)
    first_events = list(first.iter_legitimate(120))
    second_events = list(second.iter_legitimate(120))

    assert first_events == second_events
    assert first.state_summary()["events"] == 120
    assert first.state_summary()["beneficiary_payments"] == 120
    assert first.state_summary()["customers_with_history"] > 1
    assert any(event["txn"]["customer_txn_count"] > 0 for event in first_events)
    validator = Draft202012Validator(EVENT_SCHEMA, format_checker=FormatChecker())
    assert all(not list(validator.iter_errors(event)) for event in first_events)
    invalid = copy.deepcopy(
        PopulationSimulator(seed=42, customers=2, agentic_rate=1).generate_legitimate(
            force_channel="agentic_checkout"
        )
    )
    invalid["mandate_chain"]["intent"].pop("signer_public_key")
    assert list(validator.iter_errors(invalid))


def test_default_arrival_clock_follows_population_circadian_activity() -> None:
    simulator = PopulationSimulator(seed=9, customers=100, merchants_per_category=1, agentic_rate=0)
    hours = Counter(event["txn"]["hour_local"] for event in simulator.iter_legitimate(4_000))

    daytime_per_hour = sum(hours[hour] for hour in range(9, 23)) / 14
    overnight_per_hour = sum(hours[hour] for hour in range(0, 7)) / 7
    assert daytime_per_hour > 2 * overnight_per_hour


def test_calibration_profile_drives_deterministic_stateful_heterogeneity(tmp_path) -> None:
    knots = COPULA_QUANTILE_PROBABILITIES
    primitive_quantiles = {
        "invoice_count": tuple(1 + 19 * value for value in knots),
        "active_start_fraction": tuple(0.4 * value for value in knots),
        "active_duration_fraction": tuple(0.1 + 0.4 * value for value in knots),
        "relative_log_amount_center": tuple(-0.5 + value for value in knots),
        "eb_log_amount_scale": tuple(0.15 + 0.3 * value for value in knots),
        "eb_short_session_probability": tuple(0.1 + 0.4 * value for value in knots),
        "circadian_phase_latent": tuple(2 * math.pi * value for value in knots),
        "eb_circadian_concentration": (0.0,) * 32,
    }
    singleton_quantiles = {
        "active_start_fraction": tuple(0.4 * value for value in knots),
        "relative_log_amount_center": tuple(-0.5 + value for value in knots),
        "circadian_phase_latent": tuple(2 * math.pi * value for value in knots),
    }
    profile = BehaviorProfile(
        source_dataset="test aggregate",
        source_partition="calibration",
        source_customer_ids_sha256="a" * 64,
        source_transactions_sha256="b" * 64,
        source_customer_count=100,
        source_transaction_count=800,
        quantile_probabilities=knots,
        primitive_quantiles=tuple(primitive_quantiles[name] for name in COPULA_PRIMITIVES),
        singleton_quantiles=tuple(
            singleton_quantiles[name] for name in SINGLETON_PRIMITIVES
        ),
        copula_correlation=tuple(
            tuple(float(left == right) for right in range(len(COPULA_PRIMITIVES)))
            for left in range(len(COPULA_PRIMITIVES))
        ),
        hour_weights=tuple(0.5 if hour in {10, 14} else 0.0 for hour in range(24)),
        calendar_weights=(1 / 12,) * 12,
        observation_days=365,
        session_gap_log_mean=math.log(3_600),
        session_gap_log_std=0.4,
        repeat_same_day_probability=0.02,
        singleton_probability=0.25,
        circadian_phase_cut=0.0,
        amount_variance_prior_alpha=2.0,
        amount_variance_prior_beta=0.5,
        short_session_prior_alpha=1.0,
        short_session_prior_beta=3.0,
    )
    first = PopulationSimulator(
        seed=91, customers=100, merchants_per_category=1, agentic_rate=0, behavior_profile=profile
    )
    second = PopulationSimulator(
        seed=91, customers=100, merchants_per_category=1, agentic_rate=0, behavior_profile=profile
    )
    first_events = list(first.iter_legitimate(800))
    second_events = list(second.iter_legitimate(800))
    customer_counts = Counter(event["txn"]["customer_id"] for event in first_events)

    assert first_events == second_events
    assert [event["ts"] for event in first_events] == sorted(event["ts"] for event in first_events)
    assert sum(8 <= event["txn"]["hour_local"] <= 18 for event in first_events) > 0.8 * len(
        first_events
    )
    assert len(customer_counts) == 100
    assert max(customer_counts.values()) > 5 * min(customer_counts.values())
    assert any(event["txn"]["customer_txn_count"] > 1 for event in first_events)
    assert first.behavior_profile.sha256 == second.behavior_profile.sha256
    assert profile.as_dict()["source"]["partition"] == "calibration"
    assert len(profile.as_dict()["primitive_quantiles"]) == len(COPULA_PRIMITIVES)
    assert BehaviorProfile.from_dict(profile.as_dict()) == profile

    manifest = generate_population_dataset(
        tmp_path / "profiled.jsonl",
        events=900,
        seed=91,
        fraud_rate=0,
        benign_adversarial_rate=0,
        behavior_profile=profile,
    )
    assert manifest["behavior_profile_sha256"] == profile.sha256


def test_population_cli_binds_immutable_evaluation_preregistration(
    tmp_path: Path, monkeypatch
) -> None:
    preregistration = tmp_path / "evaluation_v4_preregistration.json"
    valid_preregistration = _evaluation_preregistration(ROOT)
    assert set(valid_preregistration["bindings"]) == {
        "sim_population_sha256",
        "sim_mandate_twin_sha256",
        "event_schema_sha256",
        "generator_input_tree_sha256",
        "defense_source_tree_sha256",
        "scripts_release_sha256",
        "makefile_sha256",
        "requirements_lock_sha256",
        "embedding_revision",
        "embedding_tree_sha256",
        "parent_failure_sha256",
    }
    preregistration.write_text(
        json.dumps(valid_preregistration) + "\n",
        encoding="utf-8",
    )
    preregistration.chmod(0o444)
    output = tmp_path / "events.jsonl"
    generation_calls = 0

    def fake_generate_population_dataset(path: Path, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        path.write_text("", encoding="utf-8")
        manifest = {
            "evaluation_preregistration_sha256": kwargs[
                "evaluation_preregistration_sha256"
            ]
        }
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        return manifest

    monkeypatch.setattr(
        population_module,
        "generate_population_dataset",
        fake_generate_population_dataset,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sim.population",
            "--events",
            "50000",
            "--seed",
            "20260829",
            "--evaluation-preregistration",
            str(preregistration),
            "--output",
            str(output),
        ],
    )

    population_module.main()

    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_preregistration_sha256"] == hashlib.sha256(
        preregistration.read_bytes()
    ).hexdigest()
    assert generation_calls == 1

    output.unlink()
    output.with_suffix(".manifest.json").unlink()
    for invalid_case in ("missing", "extra", "stale"):
        payload = copy.deepcopy(valid_preregistration)
        bindings = payload["bindings"]
        assert isinstance(bindings, dict)
        if invalid_case == "missing":
            bindings.pop("requirements_lock_sha256")
        elif invalid_case == "extra":
            bindings["unexpected_sha256"] = "0" * 64
        else:
            bindings["makefile_sha256"] = "0" * 64
        preregistration.chmod(0o644)
        preregistration.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        preregistration.chmod(0o444)
        with pytest.raises(RuntimeError, match="does not bind this generator"):
            population_module.main()
        assert not output.exists()
        assert generation_calls == 1


def test_agentic_legitimate_chains_verify_and_not_present_rate_is_about_65_percent() -> None:
    simulator = PopulationSimulator(seed=17, customers=80, merchants_per_category=1, agentic_rate=1)
    events = list(simulator.iter_legitimate(500))

    trusted_registry = simulator.registry.public_key_fingerprints()
    assert simulator.registry.seed == SYNTHETIC_IDENTITY_SEED
    assert all(all(verify_chain(event["mandate_chain"], trusted_registry).values()) for event in events)
    not_present = sum(event["mandate_chain"]["payment"]["presence_flag"] == "NOT_PRESENT" for event in events)
    assert 0.58 <= not_present / len(events) <= 0.72


def test_stream_base_rate_and_all_fixture_splits() -> None:
    simulator = PopulationSimulator(seed=20260827, customers=100, merchants_per_category=1, agentic_rate=0)
    frauds = sum(simulator.generate_event()["label"]["is_fraud"] for _ in range(20_000))
    assert 0.004 <= frauds / 20_000 <= 0.007

    fixture = ROOT / "sim" / "fixtures" / "events.jsonl"
    events = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    assert {event["label"]["split"] for event in events} == {"A", "B", "C", "D", "E"}
    assert all(not event["label"]["is_fraud"] for event in events if event["label"]["split"] == "E")
    assert Counter(event["label"]["attack_family"] for event in events if event["label"]["is_fraud"])

    live_default = PopulationSimulator(seed=29, customers=40, merchants_per_category=1)
    assert {live_default.generate_event(fraud_rate=1)["label"]["split"] for _ in range(100)} <= {"A", "B"}


def test_population_cli_dataset_is_deterministic_and_contains_every_split(tmp_path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = generate_population_dataset(
        first_path,
        events=240,
        seed=23,
        fraud_rate=0.05,
        benign_adversarial_rate=0.05,
    )
    second = generate_population_dataset(
        second_path,
        events=240,
        seed=23,
        fraud_rate=0.05,
        benign_adversarial_rate=0.05,
    )

    assert first["sha256"] == second["sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()
    assert set(first["split_counts"]) == {"A", "B", "C", "D", "E"}
    assert first["fraud_rate"] == 0.05
    generated = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    assert first["synthetic_identity_seed"] == SYNTHETIC_IDENTITY_SEED
    assert first["synthetic_identity_seed"] != str(23)
    assert first["key_registry"] == _observed_key_registry(generated)
    assert first["key_registry_sha256"] == key_registry_sha256(first["key_registry"])
    assert all(
        "registry_attestation" not in part
        for event in generated
        if event["mandate_chain"]
        for part in event["mandate_chain"].values()
    )
    timestamps = [event["ts"] for event in generated]
    assert timestamps == sorted(timestamps)


def test_small_fidelity_run_is_measured_not_constant(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    rng = random.Random(5)
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("amount_minor", "hour_local", "is_fraud"))
        writer.writeheader()
        for _ in range(4_000):
            writer.writerow(
                {
                    "amount_minor": round(rng.lognormvariate(11, 0.8)),
                    "hour_local": rng.choices(range(24), weights=[1 + (8 <= hour <= 22) * 4 for hour in range(24)])[0],
                    "is_fraud": False,
                }
            )
    output = tmp_path / "fidelity.json"
    result = run_fidelity(reference, output, events=1_000, seed=5)

    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert len(result["fold_auc"]) == 5
    assert math.isclose(result["auc_mean"], sum(result["fold_auc"]) / 5, abs_tol=1e-15)
    assert result["threshold_passed"] == (result["auc_mean"] <= result["threshold_max_auc"])
    assert result["gate_passed"] is False


def test_committed_200k_fidelity_measurement_is_internally_consistent() -> None:
    result = json.loads((ROOT / "results" / "fidelity.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "reference" / "creditcard_marginals.manifest.json").read_text(encoding="utf-8"))

    assert result["events_generated"] == 200_000
    assert len(result["fold_auc"]) == 5
    assert math.isclose(result["auc_mean"], sum(result["fold_auc"]) / 5, abs_tol=1e-15)
    assert result["threshold_passed"] == (result["auc_mean"] <= 0.60)
    assert result["gate_passed"] == result["threshold_passed"]
    assert result["reference"]["calibration_evaluation_disjoint"] is True
    assert result["reference"]["sha256"] == manifest["marginals_sha256"]
    assert "not behavioral or entity-state fidelity" in result["scope"]
