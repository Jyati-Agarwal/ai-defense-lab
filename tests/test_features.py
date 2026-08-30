import copy
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from defend.embeddings import CachedMiniLM
from defend.features import FEATURE_COLUMNS, FeatureExtractor
from defend.memory import PrecedentMemory


ROOT = Path(__file__).resolve().parents[1]


def test_embedding_model_load_is_strictly_offline(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def sentence_transformer(*args, **kwargs):
        called.update(args=args, kwargs=kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=sentence_transformer),
    )

    CachedMiniLM(tmp_path / "embeddings.npz")._encoder()

    assert called == {
        "args": ("sentence-transformers/all-MiniLM-L6-v2",),
        "kwargs": {"device": "cpu", "local_files_only": True},
    }


def fixture_events() -> list[dict]:
    with (ROOT / "sim" / "fixtures" / "events.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixture_registry() -> dict[str, str]:
    return json.loads(
        (ROOT / "sim" / "fixtures" / "events.manifest.json").read_text(encoding="utf-8")
    )["key_registry"]


class ExplodingEvidence(dict):
    def __getitem__(self, key):
        raise AssertionError(f"protected metadata was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"protected metadata was read: {key}")


def test_feature_names_and_runtime_are_leakage_safe(tmp_path: Path) -> None:
    events = fixture_events()
    history = [event for event in events if not event["label"]["is_fraud"] and event["mandate_chain"]][:12]
    query = copy.deepcopy(history[-1])
    query["label"] = ExplodingEvidence()
    embedder = CachedMiniLM(tmp_path / "embeddings.npz")
    memory = PrecedentMemory(embedder, tmp_path / "precedent.index", tmp_path / "precedent.json")
    memory.build(history[:-1])

    batch = FeatureExtractor(
        embedder,
        memory,
        history[:-1],
        trusted_registry=fixture_registry(),
    ).transform([query])

    forbidden = ("fraud", "atlas", "attack", "label")
    assert not [name for name in FEATURE_COLUMNS if any(token in name.lower() for token in forbidden)]
    assert tuple(batch.frame.columns[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS
    assert batch.frame.notna().all().all()


def test_deterministic_structural_floor_catches_tampering(tmp_path: Path) -> None:
    events = fixture_events()
    event = next(item for item in events if item["mandate_chain"] and not item["label"]["is_fraud"])
    tampered = copy.deepcopy(event)
    tampered["mandate_chain"]["cart"]["items"][0]["price_minor"] += 1
    expired = copy.deepcopy(event)
    intent = expired["mandate_chain"]["intent"]
    signed_at = datetime.fromisoformat(intent["signed_at"].replace("Z", "+00:00"))
    expired["ts"] = (signed_at + timedelta(seconds=int(intent["ttl_seconds"]) + 1)).isoformat().replace(
        "+00:00", "Z"
    )
    embedder = CachedMiniLM(tmp_path / "embeddings.npz")

    frame = FeatureExtractor(embedder, trusted_registry=fixture_registry()).transform(
        [event, tampered, expired]
    ).frame

    assert frame.loc[0, "structural_floor"] == 0
    assert frame.loc[1, "structural_floor"] == 1
    assert frame.loc[2, "structural_floor"] == 1


def test_credit_utilisation_regime_change_has_label_free_hold_floor(tmp_path: Path) -> None:
    def event(customer: str, prior: object, current: object, velocity: int) -> dict:
        return {
            "event_id": f"evt_{customer}",
            "ts": "2026-01-01T12:00:00Z",
            "channel": "card_ecom",
            "rail": "card",
            "mandate_chain": None,
            "agent_trace": None,
            "txn": {
                "amount_minor": 100_000,
                "currency": "INR",
                "mcc": "5411",
                "customer_id": customer,
                "device_id": f"dev_{customer}",
                "geo": {"lat": 19.076, "lon": 72.8777},
                "hour_local": 12,
                "beneficiary_id": f"ben_{customer}",
                "is_new_beneficiary": False,
                "prior_utilisation_pct": prior,
                "current_utilisation_pct": current,
                "cashout_velocity_1h": velocity,
            },
            "label": ExplodingEvidence(),
        }

    rows = [
        event("trigger", 0.08, 0.98, 11),
        event("boundary", 0.20, 0.90, 5),
        event("stable_high", 0.92, 0.98, 11),
        event("not_exhausted", 0.08, 0.88, 11),
        event("paced", 0.08, 0.98, 4),
        event("invalid", 0.08, 1.01, 11),
        event("not_finite", 0.08, float("nan"), 11),
    ]

    frame = FeatureExtractor(CachedMiniLM(tmp_path / "embeddings.npz")).transform(rows).frame

    assert frame["credit_utilisation_regime_change"].tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert frame["structural_floor"].tolist() == [0.60, 0.60, 0, 0, 0, 0, 0]
