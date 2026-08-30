import json
import os
from pathlib import Path

import numpy as np
import pytest

import loop.arena as arena_module
from defend.embeddings import CachedMiniLM
from defend.harness import make_split_plan, model_hash
from defend.memory import PrecedentMemory
from defend.train import EVENT_PROVENANCE_FIELDS, load_events, load_events_with_provenance, train_events
from loop.evasion import fitness


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("AI_DEFENSE_TEST_RESULTS", ROOT / "results"))


def test_five_round_arena_is_explicitly_in_distribution() -> None:
    arena = json.loads((RESULTS / "arena_log.json").read_text(encoding="utf-8"))
    phylogeny = json.loads((RESULTS / "phylogeny.json").read_text(encoding="utf-8"))

    assert len(arena["rounds"]) == 5
    assert [row["round"] for row in arena["rounds"]] == [1, 2, 3, 4, 5]
    assert arena["training_partitions"] == ["A", "B"]
    assert arena["input_partitions_materialized"] == ["A", "B", "C", "D", "E"]
    assert arena["partitions_used_for_training_or_feature_extraction"] == ["A", "B"]
    assert "evaluation_partitions_seen" not in arena
    assert "not generalization" in arena["label"].lower()
    assert arena["rounds"][-1]["selected_training_count"] == 0
    assert arena["rounds"][-1]["disposition"] == "reserved for guarded vaccination holdout"
    assert all("adaptive_asr" in row for row in arena["rounds"])
    assert arena["hardening_gate"]["passed"] is True
    assert arena["rounds"][-1]["asr"] == 0 or arena["rounds"][-1]["asr"] < arena["rounds"][0]["asr"]
    assert arena["replay_challenge"]["count"] == 48
    assert arena["replay_challenge"]["source_partitions"] == ["A", "B"]
    assert arena["replay_challenge"]["training_included"] is False
    assert arena["reserved_final_round_survivors"] >= 2
    assert any(node.get("emergent") for node in phylogeny["nodes"])
    assert (RESULTS / "asr_curve.png").stat().st_size > 1_000


def test_evasion_fitness_rewards_payoff_and_lower_blue_score() -> None:
    assert fitness(100, 0.1) > fitness(100, 0.9)
    assert fitness(200, 0.5) > fitness(100, 0.5)


def test_failed_replay_gate_preserves_model_and_arena_artifacts(tmp_path: Path, monkeypatch) -> None:
    rows, provenance = load_events_with_provenance(ROOT / "sim" / "fixtures" / "events.jsonl")
    train_events(rows, tmp_path, events_provenance=provenance)
    protected = {
        "model.pkl": (tmp_path / "model.pkl").read_bytes(),
        "arena_log.json": b'{"sentinel":"arena"}\n',
        "arena_survivors.jsonl": b'{"sentinel":"survivors"}\n',
        "phylogeny.json": b'{"sentinel":"phylogeny"}\n',
        "asr_curve.png": b"sentinel-plot",
        "arena_release.json": b'{"sentinel":"release"}\n',
    }
    for name, content in protected.items():
        if name != "model.pkl":
            (tmp_path / name).write_bytes(content)
    base_hash = model_hash(tmp_path / "model.pkl")

    monkeypatch.setattr(
        arena_module,
        "_fused_scores",
        lambda _model, batch: np.zeros(len(batch.frame), dtype=float),
    )
    with pytest.raises(AssertionError, match="neither improved nor reached zero"):
        arena_module.run_arena(
            rows, tmp_path, use_cache=False, events_provenance=provenance
        )

    assert model_hash(tmp_path / "model.pkl") == base_hash
    for name, content in protected.items():
        assert (tmp_path / name).read_bytes() == content
    assert not list(tmp_path.glob(".arena-staging-*"))


def test_zero_replay_asr_is_a_valid_floor(tmp_path: Path, monkeypatch) -> None:
    rows, provenance = load_events_with_provenance(ROOT / "sim" / "fixtures" / "events.jsonl")
    train_events(rows, tmp_path, events_provenance=provenance)
    poisoned = next(event for event in rows if event["label"]["split"] == "C" and event.get("mandate_chain"))
    PrecedentMemory(
        CachedMiniLM(tmp_path / "embeddings.npz"),
        tmp_path / "precedent.index",
        tmp_path / "precedent_history.json",
    ).build([poisoned])
    monkeypatch.setattr(
        arena_module,
        "_fused_scores",
        lambda _model, batch: np.ones(len(batch.frame), dtype=float),
    )

    arena = arena_module.run_arena(
        rows, tmp_path, use_cache=False, events_provenance=provenance
    )

    assert arena["seed"] == 20260827
    assert arena["hardening_gate"] == {
        "metric": "fixed_replay_attack_success_rate",
        "baseline": 0.0,
        "final": 0.0,
        "operator": "final == 0 or final < baseline",
        "passed": True,
        "outcome": "zero_residual_floor",
    }
    expected = [str(event["event_id"]) for event in make_split_plan(rows).precedent_history]
    records = json.loads((tmp_path / "precedent_history.json").read_text(encoding="utf-8"))
    assert [record["event_id"] for record in records] == expected


def test_arena_rejects_training_manifest_from_different_inputs(tmp_path: Path) -> None:
    rows, provenance = load_events_with_provenance(ROOT / "sim" / "fixtures" / "events.jsonl")
    train_events(rows, tmp_path, events_provenance=provenance)
    manifest_path = tmp_path / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_fields = (
        "dataset_sha256",
        "split_plan_sha256",
        "source_tree_sha256",
        *EVENT_PROVENANCE_FIELDS,
    )
    assert {key: manifest[key] for key in EVENT_PROVENANCE_FIELDS} == {
        key: provenance[key] for key in EVENT_PROVENANCE_FIELDS
    }
    for key in evidence_fields:
        manifest[key] = f"wrong-{key}"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    base_hash = model_hash(tmp_path / "model.pkl")

    with pytest.raises(RuntimeError) as failure:
        arena_module.run_arena(
            rows,
            tmp_path,
            use_cache=False,
            events_provenance=provenance,
        )

    message = str(failure.value)
    for key in evidence_fields:
        assert key in message
    assert model_hash(tmp_path / "model.pkl") == base_hash
