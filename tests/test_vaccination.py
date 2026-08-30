import copy
import json
import os
from pathlib import Path

import joblib
import pytest

import defend.vaccination as vaccination_module
from defend.embeddings import CachedMiniLM
from defend.harness import canonical_events_hash, make_split_plan, model_hash, target
from defend.memory import PrecedentMemory
from defend.train import load_events_with_provenance, train_events


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("AI_DEFENSE_TEST_RESULTS", ROOT / "results"))


def test_guarded_vaccination_does_not_mutate_live_model() -> None:
    artifact = json.loads((RESULTS / "vaccination.json").read_text(encoding="utf-8"))
    generalization = json.loads((RESULTS / "generalization.json").read_text(encoding="utf-8"))

    assert artifact["live_model_unchanged"] is True
    assert artifact["live_model_sha256"] == model_hash(RESULTS / "model.pkl")
    assert model_hash(RESULTS / "model.pkl") == generalization["model_hash"]
    assert [stage["name"] for stage in artifact["stages"]] == [
        "Candidate mini-dataset + shallow rule",
        "Reserved survivor holdout",
        "Old-path regression",
        "Benign E regression",
        "Calibration check",
        "Human promotion boundary",
    ]
    assert artifact["checks"]["survivor_stop_rate_after"] > artifact["checks"]["survivor_stop_rate_before"]
    assert artifact["checks"]["old_stop_rate_after"] >= artifact["checks"]["old_stop_rate_before"]
    assert artifact["checks"]["benign_e_stop_rate_after"] > artifact["checks"]["benign_e_stop_rate_before"]
    assert artifact["checks"]["brier_after"] > artifact["checks"]["brier_before"]
    assert [stage["status"] for stage in artifact["stages"][-3:]] == ["rejected"] * 3
    assert artifact["promoted_artifact"] is None
    if artifact["candidate_artifact"]:
        candidate = joblib.load(RESULTS / artifact["candidate_artifact"])
        assert candidate["status"] == "awaiting_human_review"
        assert candidate["base_model_sha256"] == model_hash(RESULTS / "model.pkl")


def test_vaccination_replaces_stale_precedent_cache(tmp_path: Path) -> None:
    rows, provenance = load_events_with_provenance(
        ROOT / "sim" / "fixtures" / "events.jsonl"
    )
    run = train_events(rows, tmp_path, events_provenance=provenance)
    fraudulent = next(event for event in run.plan.training if target(event))
    survivors = []
    for number in range(3):
        survivor = copy.deepcopy(fraudulent)
        survivor["event_id"] = f"fixture-survivor-{number}"
        survivors.append(survivor)
    survivor_path = tmp_path / "arena_survivors.jsonl"
    survivor_path.write_text(
        "".join(json.dumps(event) + "\n" for event in survivors),
        encoding="utf-8",
    )
    arena_path = tmp_path / "arena_log.json"
    arena_path.write_text(
        json.dumps(
            {
                "seed": 20260827,
                "rounds": [{"mutation": "synthetic fixture"}],
                "hardening_gate": {"passed": True},
                "dataset_sha256": canonical_events_hash(rows),
                "split_plan_sha256": run.plan.digest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    live_hash = model_hash(tmp_path / "model.pkl")
    (tmp_path / "arena_release.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "seed": 20260827,
                "final_model_hash": live_hash,
                "dataset_sha256": canonical_events_hash(rows),
                "split_plan_sha256": run.plan.digest(),
                "artifact_sha256": {
                    "model.pkl": live_hash,
                    "arena_log.json": model_hash(arena_path),
                    "arena_survivors.jsonl": model_hash(survivor_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    poisoned = next(event for event in rows if event["label"]["split"] == "C" and event.get("mandate_chain"))
    PrecedentMemory(
        CachedMiniLM(tmp_path / "embeddings.npz"),
        tmp_path / "precedent.index",
        tmp_path / "precedent_history.json",
    ).build([poisoned])

    altered = copy.deepcopy(rows)
    altered[0]["txn"]["amount_minor"] += 1
    with pytest.raises(RuntimeError, match="exact passed arena inputs"):
        vaccination_module.run_vaccination(
            altered, tmp_path, events_provenance=provenance
        )

    vaccination_module.run_vaccination(
        rows, tmp_path, events_provenance=provenance
    )

    expected = [str(event["event_id"]) for event in make_split_plan(rows).precedent_history]
    records = json.loads((tmp_path / "precedent_history.json").read_text(encoding="utf-8"))
    assert [record["event_id"] for record in records] == expected
