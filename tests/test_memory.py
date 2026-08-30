import copy
import json
from pathlib import Path

from defend.embeddings import CachedMiniLM
from defend.features import FeatureExtractor
from defend.memory import PrecedentMemory


ROOT = Path(__file__).resolve().parents[1]


def test_faiss_precedent_support_distinguishes_legitimate_context(tmp_path: Path) -> None:
    with (ROOT / "sim" / "fixtures" / "events.jsonl").open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    base = next(event for event in events if event["mandate_chain"] and not event["label"]["is_fraud"])
    history = [copy.deepcopy(base) for _ in range(6)]
    for number, event in enumerate(history):
        event["event_id"] = f"evt_history_{number}"
    unusual = copy.deepcopy(base)
    unusual["event_id"] = "evt_unusual"
    unusual["mandate_chain"]["cart"]["items"].append(
        {"sku": "HYDRATION-SAFE", "name": "Hydration vest", "price_minor": 10_000, "qty": 1}
    )
    injected = copy.deepcopy(base)
    injected["event_id"] = "evt_injected"
    injected["mandate_chain"]["cart"]["items"] = [
        {"sku": "WALLET-SAFE", "name": "Crypto wallet voucher", "price_minor": 10_000, "qty": 1}
    ]
    embedder = CachedMiniLM(tmp_path / "embeddings.npz")
    memory = PrecedentMemory(embedder, tmp_path / "precedent.index", tmp_path / "precedent.json")
    memory.build(history)

    legitimate_support = memory.support(unusual)
    injected_support = memory.support(injected)

    assert memory.index is not None
    assert legitimate_support > injected_support
    assert legitimate_support > 0.75
    trusted_registry = json.loads(
        (ROOT / "sim" / "fixtures" / "events.manifest.json").read_text(encoding="utf-8")
    )["key_registry"]
    frame = FeatureExtractor(
        embedder,
        memory,
        history,
        trusted_registry=trusted_registry,
    ).transform([unusual, injected]).frame
    assert frame.loc[0, "contextualized_semantic_drift"] < frame.loc[0, "intent_cart_semantic_drift"]
    assert frame.loc[1, "contextualized_semantic_drift"] > frame.loc[0, "contextualized_semantic_drift"]
