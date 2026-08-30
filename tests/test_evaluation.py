import inspect
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import defend.evaluate as evaluate_module
import defend.train as train_module
import loop.arena as arena_module
from defend.embeddings import CachedMiniLM
from defend.evaluate import _average_precision_fixed_ranking, _roc_auc_fixed_ranks, _score_ranks, evaluate_events
from defend.features import FeatureBatch
from defend.harness import (
    FrozenPartitionScorer,
    canonical_events_hash,
    make_split_plan,
    model_hash,
    source_tree_hash,
)
from defend.memory import PrecedentMemory
from defend.train import (
    event_provenance,
    load_events,
    load_events_with_provenance,
    train_events,
    trusted_registry_from_provenance,
)
from sim.mandate_twin import key_registry_sha256
from tests.test_release import _evaluation_preregistration


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("AI_DEFENSE_TEST_RESULTS", ROOT / "results"))
EVENTS = ROOT / "data" / "events.jsonl"


def _bind_test_preregistration(
    output: Path, provenance: dict[str, str | None]
) -> dict[str, str | None]:
    path = output / evaluate_module.EVALUATION_PREREGISTRATION
    path.write_text(
        json.dumps(_evaluation_preregistration(ROOT)) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return {**provenance, "evaluation_preregistration_sha256": model_hash(path)}


def test_temporal_split_is_global_and_training_is_a_b_only() -> None:
    plan = make_split_plan(load_events(EVENTS))
    assert max(event["ts"] for event in plan.precedent_history + plan.training) < min(
        event["ts"] for event in plan.evaluation()
    )
    assert plan.temporal_cutoff == min(event["ts"] for event in plan.evaluation())
    assert {event["label"]["split"] for event in plan.training} <= {"A", "B"}


def test_measured_frozen_evaluation_artifacts() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    generalization = json.loads((RESULTS / "generalization.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "frozen_scoring.json").read_text(encoding="utf-8"))
    scored = [json.loads(line) for line in (RESULTS / "scored_events.jsonl").read_text(encoding="utf-8").splitlines()]

    assert generalization["model_hash"] == model_hash(RESULTS / "model.pkl") == audit["model_hash"]
    assert audit["partitions_scored_once"] == ["C", "D", "E"]
    checks = generalization["contamination_checks"]
    assert len(checks["shuffled_label_pr_aucs"]) == 50
    assert checks["shuffled_label_summary"].startswith("mean-score ensemble of 50 full DefenseModel")
    assert checks["shuffled_label_gate"] == "PASS"
    assert checks["empirical_permutations"] == 9_999
    assert checks["empirical_upper_tail_p"] >= 0.01
    assert checks["control_scope"].startswith("temporal A has-text stratum only")
    assert checks["protocol_id"] == "temporal-a-has-text-conditional-v2-2026-08-27"
    assert checks["modality_shortcut_audit"]["all_positives_retained_in_v2"] is True
    assert checks["v1_mixed_temporal_a"]["status"] == "PASS"
    assert checks["v1_mixed_temporal_a"]["certification_use"] == "preserved diagnostic only"
    assert generalization["headline"] == "heldout_family_recall"
    assert all(event["decision"]["reason_codes"] for event in scored)
    assert len(scored) == sum(generalization["scoring_counts"].values())
    assert metrics["ablation_table"] and metrics["pr_curve"] and metrics["per_attack_id_recall"]
    assert metrics["latency"]["p99_ms"] < 100
    assert metrics["latency"]["latency_benchmark_scope"] == "A+B"
    assert metrics["latency"]["protected_partition_inference_passes"] == 1
    assert audit["latency_benchmark_scope"] == "A+B"
    assert audit["protected_partition_inference_passes"] == 1
    assert all(row["pr_auc_drop"] > 0 or row["heldout_recall_drop"] > 0 for row in metrics["ablation_table"][1:])
    assert all(value > 0 for value in metrics["per_attack_id_recall"].values())
    assert generalization["recall_evidence"]["heldout_family"]["positive_examples"] > 0
    for name in ("pr_curve.png", "attack_recall_heatmap.png"):
        assert (RESULTS / name).stat().st_size > 1_000

    release_path = RESULTS / evaluate_module.EVALUATION_RELEASE
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert release["status"] == "passed"
    assert stat.S_IMODE(release_path.stat().st_mode) == 0o444
    assert all(
        model_hash(RESULTS / name) == digest
        for name, digest in release["artifact_sha256"].items()
    )
    scorer = FrozenPartitionScorer()
    scorer.claim("C")
    with pytest.raises(RuntimeError):
        scorer.claim("C")


def test_fixed_ranking_average_precision_matches_sklearn_with_ties() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=int)
    scores = np.asarray([0.8, 0.8, 0.4, 0.2, 0.2, 0.1], dtype=float)

    assert _average_precision_fixed_ranking(labels, scores) == pytest.approx(
        average_precision_score(labels, scores)
    )
    assert _roc_auc_fixed_ranks(labels, _score_ranks(scores)) == pytest.approx(roc_auc_score(labels, scores))


def test_latency_benchmark_never_repredicts_protected_partitions() -> None:
    def partition_batch(name: str) -> FeatureBatch:
        return FeatureBatch(
            pd.DataFrame({"partition_marker": [name]}),
            np.zeros((1, 2), dtype=np.float32),
            np.asarray([False], dtype=bool),
            [{"partition": name}],
        )

    partition_batches = {
        name: partition_batch(name) for name in ("A", "B", "C", "D", "E")
    }
    benchmark_batch = evaluate_module._latency_benchmark_batch(partition_batches)
    observed: list[str] = []

    class RecordingModel:
        def predict(self, batch: FeatureBatch) -> dict:
            observed.extend(batch.frame["partition_marker"].tolist())
            return {}

    evaluate_module._benchmark_singleton_latency(
        RecordingModel(), benchmark_batch, sample_count=5
    )

    assert set(observed) == {"A", "B"}
    assert not ({"C", "D", "E"} & set(observed))


def _control_batch(signal: list[float], has_text: list[bool]) -> FeatureBatch:
    frame = pd.DataFrame({
        "signal": signal,
        "structural_floor": np.ones(len(signal)),
        "layer_semantic": np.ones(len(signal)),
        "layer_provenance": np.ones(len(signal)),
        "layer_temporal": np.ones(len(signal)),
        "layer_economic": np.ones(len(signal)),
    })
    return FeatureBatch(
        frame,
        np.zeros((len(signal), 2), dtype=np.float32),
        np.asarray(has_text, dtype=bool),
        [{} for _ in signal],
    )


def test_conditional_shuffle_preserves_v1_and_requires_all_positives(monkeypatch) -> None:
    class FakeFusion:
        @staticmethod
        def predict_proba(layers):
            return np.asarray(layers[:, 1], dtype=float)

    class FakeDefenseModel:
        def __init__(self, seed):
            self.seed = seed
            self.fusion = FakeFusion()

        def fit(self, _batch, _targets):
            return self

        @staticmethod
        def layer_scores(batch):
            signal = batch.frame["signal"].to_numpy(dtype=float)
            return np.column_stack([np.zeros(len(signal)), signal, np.zeros((len(signal), 3))])

    monkeypatch.setattr(evaluate_module, "DefenseModel", FakeDefenseModel)
    training = _control_batch([0.1, 0.2, 0.3, 0.4], [True, True, False, False])
    training_targets = np.asarray([0, 1, 0, 1], dtype=int)
    evaluation = _control_batch(
        [0.9, 0.8, 0.1, 0.2, 0.99, 0.98, 0.97, 0.96],
        [True, True, True, True, False, False, False, False],
    )
    evaluation_targets = np.asarray([1, 0, 1, 0, 0, 0, 0, 0], dtype=int)

    control = evaluate_module._shuffled_label_control(
        training,
        training_targets,
        evaluation,
        evaluation_targets,
        seed=31,
    )

    assert len(control["records"]) == 50
    assert control["v1_mixed_temporal_a"]["certification_use"] == "preserved diagnostic only"
    assert control["v1_mixed_temporal_a"]["ensemble"]["pr_auc"] == pytest.approx(
        average_precision_score(evaluation_targets, evaluation.frame["signal"])
    )
    conditional = evaluation.has_text
    assert control["ensemble"]["pr_auc"] == pytest.approx(
        average_precision_score(evaluation_targets[conditional], evaluation.frame.loc[conditional, "signal"])
    )
    assert control["gate"]["scope"] == "temporal-A has_text=True, all positives retained"
    assert control["modality_shortcut_audit"]["all_positives_retained_in_v2"] is True

    missing_positive = _control_batch([0.9, 0.8, 0.1], [False, True, True])
    with pytest.raises(AssertionError, match="retain every temporal-A positive"):
        evaluate_module._conditional_has_text_mask(
            missing_positive,
            np.asarray([1, 0, 0], dtype=int),
        )


def test_cli_orchestration_has_one_safe_order(monkeypatch, capsys) -> None:
    evaluate_source = inspect.getsource(evaluate_module.main)
    train_source = inspect.getsource(train_module.main)

    assert "run_arena" not in evaluate_source
    assert "run_arena" not in train_source
    assert "evaluate_events" not in train_source
    assert "run_vaccination" not in train_source
    monkeypatch.setattr(sys, "argv", ["defend.train", "--all"])
    with pytest.raises(SystemExit) as failure:
        train_module.main()
    assert failure.value.code == 2
    assert "--all was removed; use `make all`" in capsys.readouterr().err


def test_event_manifest_validation_and_cli_enforcement(tmp_path: Path, monkeypatch) -> None:
    fixture_events = ROOT / "sim" / "fixtures" / "events.jsonl"
    fixture_manifest = ROOT / "sim" / "fixtures" / "events.manifest.json"
    events_path = tmp_path / "events.jsonl"
    manifest_path = tmp_path / "events.manifest.json"
    events_path.write_bytes(fixture_events.read_bytes())

    for program, main in (
        ("defend.train", train_module.main),
        ("loop.arena", arena_module.main),
        ("defend.evaluate", evaluate_module.main),
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [program, "--events", str(events_path), "--results", str(tmp_path / "results")],
        )
        with pytest.raises(FileNotFoundError, match="adjacent manifest"):
            main()

    original_manifest = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    profile_hash = "c" * 64
    valid_manifest = {
        **original_manifest,
        "behavior_profile_sha256": profile_hash,
        "evaluation_preregistration_sha256": "d" * 64,
    }
    manifest_path.write_text(json.dumps(valid_manifest) + "\n", encoding="utf-8")
    rows, provenance = load_events_with_provenance(events_path)
    assert len(rows) == original_manifest["events"]
    assert provenance == {
        "events_file_sha256": model_hash(events_path),
        "events_manifest_sha256": model_hash(manifest_path),
        "behavior_profile_sha256": profile_hash,
        "evaluation_preregistration_sha256": "d" * 64,
        "key_registry_sha256": original_manifest["key_registry_sha256"],
        "key_registry": original_manifest["key_registry"],
    }

    partial_registry = dict(original_manifest["key_registry"])
    partial_registry.pop(next(iter(partial_registry)))
    partial_manifest = {
        **valid_manifest,
        "key_registry": partial_registry,
        "key_registry_sha256": key_registry_sha256(partial_registry),
    }
    manifest_path.write_text(json.dumps(partial_manifest) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly cover"):
        load_events_with_provenance(events_path)

    substituted_provenance = {
        **provenance,
        "key_registry": partial_registry,
        "key_registry_sha256": key_registry_sha256(partial_registry),
    }
    with pytest.raises(RuntimeError, match="exactly cover"):
        trusted_registry_from_provenance(substituted_provenance, rows)

    count_mismatch = {**valid_manifest, "events": len(rows) + 1}
    manifest_path.write_text(json.dumps(count_mismatch) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="count mismatch"):
        load_events_with_provenance(events_path)

    sha_mismatch = {**valid_manifest, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(sha_mismatch) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        load_events_with_provenance(events_path)


def _write_valid_arena_release(
    rows: list[dict],
    output: Path,
    provenance: dict[str, str | None] | None = None,
) -> None:
    plan = make_split_plan(rows)
    artifact_names = (
        "model.pkl",
        "arena_survivors.jsonl",
        "arena_log.json",
        "phylogeny.json",
        "asr_curve.png",
    )
    for name in artifact_names[1:]:
        (output / name).write_bytes(f"fixture-{name}".encode())
    final_hash = model_hash(output / "model.pkl")
    release = {
        "status": "passed",
        "seed": 20260827,
        "final_model_hash": final_hash,
        "dataset_sha256": canonical_events_hash(rows),
        "split_plan_sha256": plan.digest(),
        "source_tree_sha256": source_tree_hash(ROOT),
        **event_provenance(provenance),
        "artifact_sha256": {name: model_hash(output / name) for name in artifact_names},
    }
    (output / "arena_release.json").write_text(json.dumps(release) + "\n", encoding="utf-8")


def test_failed_shuffle_gate_never_opens_frozen_partitions_or_promotes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows, provenance = load_events_with_provenance(ROOT / "sim" / "fixtures" / "events.jsonl")
    provenance = _bind_test_preregistration(tmp_path, provenance)
    train_events(rows, tmp_path, events_provenance=provenance)
    with pytest.raises(FileNotFoundError, match="passed arena release"):
        evaluate_events(rows, tmp_path, events_provenance=provenance)
    prerequisite_failure = json.loads(
        (tmp_path / evaluate_module.EVALUATION_RELEASE).read_text(encoding="utf-8")
    )
    assert prerequisite_failure["status"] == "failed"
    bound_provenance = event_provenance(provenance)
    assert {key: prerequisite_failure[key] for key in bound_provenance} == bound_provenance
    _write_valid_arena_release(rows, tmp_path, provenance)
    poisoned = next(event for event in rows if event["label"]["split"] == "C" and event.get("mandate_chain"))
    PrecedentMemory(
        CachedMiniLM(tmp_path / "embeddings.npz"),
        tmp_path / "precedent.index",
        tmp_path / "precedent_history.json",
    ).build([poisoned])
    protected = {
        "metrics.json": b'{"sentinel":"metrics"}\n',
        "generalization.json": b'{"sentinel":"generalization"}\n',
        "scored_events.jsonl": b'{"sentinel":"scored"}\n',
        "shuffle_control.json": b'{"sentinel":"shuffle"}\n',
        "frozen_scoring.json": b'{"model_hash":"stale-sentinel"}\n',
        "pr_curve.png": b"sentinel-pr",
        "attack_recall_heatmap.png": b"sentinel-recall",
    }
    for name, content in protected.items():
        (tmp_path / name).write_bytes(content)

    original_batch_for = evaluate_module._batch_for

    def guarded_batch_for(extractor, events):
        assert {event["label"]["split"] for event in events} <= {"A"}
        return original_batch_for(extractor, events)

    monkeypatch.setattr(evaluate_module, "_batch_for", guarded_batch_for)
    monkeypatch.setattr(
        evaluate_module,
        "_shuffled_label_control",
        lambda *_args, **_kwargs: {
            "gate": {"passed": False},
            "ensemble": {
                "pr_auc": 0.2,
                "pr_auc_null_upper_99": 0.1,
                "pr_auc_upper_tail_p": 0.001,
            },
        },
    )

    def unexpected_claim(_self, name):
        raise AssertionError(f"frozen partition {name} was opened before the shuffle gate")

    monkeypatch.setattr(evaluate_module.FrozenPartitionScorer, "claim", unexpected_claim)
    with pytest.raises(AssertionError, match="retained above-chance"):
        evaluate_events(rows, tmp_path, events_provenance=provenance)

    for name, content in protected.items():
        assert (tmp_path / name).read_bytes() == content
    release = json.loads(
        (tmp_path / evaluate_module.EVALUATION_RELEASE).read_text(encoding="utf-8")
    )
    assert release["status"] == "failed"
    assert "shuffled-label" in release["reason"]
    bound_provenance = event_provenance(provenance)
    assert {key: release[key] for key in bound_provenance} == bound_provenance
    assert not list(tmp_path.glob(".evaluation-staging-*"))
    expected = [str(event["event_id"]) for event in make_split_plan(rows).precedent_history]
    records = json.loads((tmp_path / "precedent_history.json").read_text(encoding="utf-8"))
    assert [record["event_id"] for record in records] == expected


def test_one_shot_receipt_is_written_before_frozen_scoring_and_blocks_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows, provenance = load_events_with_provenance(ROOT / "sim" / "fixtures" / "events.jsonl")
    provenance = _bind_test_preregistration(tmp_path, provenance)
    legacy_release_path = tmp_path / "evaluation_release.json"
    legacy_release = b'{"status":"failed","reason":"immutable V3 sentinel"}\n'
    legacy_release_path.write_bytes(legacy_release)
    legacy_release_path.chmod(0o444)
    train_events(rows, tmp_path, events_provenance=provenance)
    _write_valid_arena_release(rows, tmp_path, provenance)
    monkeypatch.setattr(
        evaluate_module,
        "_shuffled_label_control",
        lambda *_args, **_kwargs: {"gate": {"passed": True}},
    )
    attempt_path = tmp_path / evaluate_module.EVALUATION_ATTEMPT

    def fail_after_receipt(_self, name):
        assert name == "C"
        assert attempt_path.is_file()
        assert attempt_path.stat().st_mode & 0o222 == 0
        raise RuntimeError("forced post-receipt failure")

    monkeypatch.setattr(evaluate_module.FrozenPartitionScorer, "claim", fail_after_receipt)
    with pytest.raises(RuntimeError, match="forced post-receipt failure"):
        evaluate_events(rows, tmp_path, events_provenance=provenance)

    receipt = attempt_path.read_bytes()
    receipt_payload = json.loads(receipt)
    assert receipt_payload["schema_version"] == 3
    assert receipt_payload["artifact"] == "evaluation_attempt_v3"
    assert receipt_payload["claim_scope"] == "post_remediation_regression"
    assert receipt_payload["fresh_rows_only"] is True
    assert receipt_payload["independent_holdout"] is False
    assert receipt_payload["pristine_heldout_family"] is False
    assert receipt_payload["prior_families_disclosed"] is True
    assert receipt_payload["disclosed_vector_ids"] == ["GPF-A20", "GPF-C04"]
    assert receipt_payload["disclosed_family_ids"] == [
        "synthetic_identity_bust_out",
        "ttl_dormancy",
    ]
    assert stat.S_IMODE(attempt_path.stat().st_mode) == 0o444
    release = json.loads(
        (tmp_path / evaluate_module.EVALUATION_RELEASE).read_text(encoding="utf-8")
    )
    assert release["schema_version"] == 4
    assert release["artifact"] == "evaluation_v4_release"
    assert release["evaluation_scope"] == evaluate_module.EVALUATION_SCOPE
    assert release["status"] == "failed"
    assert release["attempt_receipt"] == evaluate_module.EVALUATION_ATTEMPT
    assert legacy_release_path.read_bytes() == legacy_release
    assert stat.S_IMODE(legacy_release_path.stat().st_mode) == 0o444

    with pytest.raises(RuntimeError, match="already consumed"):
        evaluate_events(rows, tmp_path, events_provenance=provenance)
    assert attempt_path.read_bytes() == receipt


def test_non_preregistered_seed_is_rejected_before_attempt(
    tmp_path: Path,
) -> None:
    rows, provenance = load_events_with_provenance(ROOT / "sim" / "fixtures" / "events.jsonl")
    provenance = _bind_test_preregistration(tmp_path, provenance)
    train_events(rows, tmp_path, events_provenance=provenance)

    with pytest.raises(RuntimeError, match="preregistration does not match"):
        evaluate_events(rows, tmp_path, seed=1, events_provenance=provenance)

    assert not (tmp_path / evaluate_module.EVALUATION_ATTEMPT).exists()


def test_incomplete_preregistration_is_rejected_before_attempt(tmp_path: Path) -> None:
    rows, provenance = load_events_with_provenance(
        ROOT / "sim" / "fixtures" / "events.jsonl"
    )
    preregistration = _evaluation_preregistration(ROOT)
    custody = preregistration["custody"]
    assert isinstance(custody, dict)
    custody.pop("retry_after_consumption")
    path = tmp_path / evaluate_module.EVALUATION_PREREGISTRATION
    path.write_text(json.dumps(preregistration) + "\n", encoding="utf-8")
    path.chmod(0o444)
    provenance = {
        **provenance,
        "evaluation_preregistration_sha256": model_hash(path),
    }
    train_events(rows, tmp_path, events_provenance=provenance)

    def protected_events():
        raise AssertionError("protected partitions opened before preregistration")
        yield {}

    with pytest.raises(RuntimeError, match="preregistration does not match"):
        evaluate_module._evaluate_events(
            protected_events(), tmp_path, events_provenance=provenance
        )

    assert not (tmp_path / evaluate_module.EVALUATION_ATTEMPT).exists()


def test_rejected_invocation_cannot_overwrite_a_passed_release(
    tmp_path: Path, monkeypatch
) -> None:
    release_path = tmp_path / evaluate_module.EVALUATION_RELEASE
    passed = b'{"status":"passed","sentinel":"last-good"}\n'
    release_path.write_bytes(passed)

    def reject(*_args, **_kwargs):
        release_path.write_text('{"status":"failed"}\n', encoding="utf-8")
        raise RuntimeError("rejected invocation")

    monkeypatch.setattr(evaluate_module, "_evaluate_events", reject)
    with pytest.raises(RuntimeError, match="rejected invocation"):
        evaluate_events([], tmp_path)

    assert release_path.read_bytes() == passed
