"""FAISS memory of verified historical intent-to-purchase pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from defend.embeddings import CachedMiniLM


def intent_text(event: dict[str, Any]) -> str:
    chain = event.get("mandate_chain") or {}
    intent = chain.get("intent") or {}
    return str(intent.get("prompt_playback") or intent.get("category") or "")


def purchase_text(event: dict[str, Any]) -> str:
    chain = event.get("mandate_chain") or {}
    cart = chain.get("cart") or {}
    items = cart.get("items") or []
    return " + ".join(str(item.get("name") or item.get("sku") or "") for item in items).strip(" +")


def pair_text(event: dict[str, Any]) -> str:
    return f"Requested: {intent_text(event)}\nPurchased: {purchase_text(event)}"


def _pair_vectors(embedder: CachedMiniLM, events: list[dict[str, Any]]) -> np.ndarray:
    intents = embedder.encode(intent_text(event) for event in events)
    purchases = embedder.encode(purchase_text(event) for event in events)
    return np.concatenate((intents, purchases), axis=1).astype(np.float32, copy=False)


class PrecedentMemory:
    """Exact cosine search over separately held-back, verified legitimate history."""

    def __init__(
        self,
        embedder: CachedMiniLM,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        self.embedder = embedder
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.Index | None = None
        self.records: list[dict[str, str]] = []
        if self.index_path.is_file() and self.metadata_path.is_file():
            self.index = faiss.read_index(str(self.index_path))
            self.records = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def build(self, verified_history: Iterable[dict[str, Any]]) -> None:
        rows = [event for event in verified_history if intent_text(event) and purchase_text(event)]
        if not rows:
            raise ValueError("precedent history has no intent-to-purchase pairs")
        vectors = _pair_vectors(self.embedder, rows)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.records = [
            {
                "event_id": str(event.get("event_id", "historical")),
                "intent": intent_text(event),
                "purchase": purchase_text(event),
            }
            for event in rows
        ]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.embedder.flush()

    def search_many(self, events: Iterable[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
        rows = list(events)
        if self.index is None or not self.records:
            return [{"support": 0.0, "neighbors": []} for _ in rows]
        query = _pair_vectors(self.embedder, rows)
        faiss.normalize_L2(query)
        count = min(max(1, k), len(self.records))
        similarities, positions = self.index.search(query, count)
        output = []
        for row_similarities, row_positions in zip(similarities, positions, strict=True):
            neighbors = []
            for similarity, position in zip(row_similarities, row_positions, strict=True):
                if position < 0:
                    continue
                record = self.records[int(position)]
                neighbors.append(
                    {
                        "title": record["intent"],
                        "similarity": round(float(np.clip(similarity, -1, 1)), 6),
                        "outcome": "Verified legitimate",
                        "context": record["purchase"],
                        "event_id": record["event_id"],
                    }
                )
            positive = [max(0.0, item["similarity"]) for item in neighbors]
            support = 0.0 if not positive else 0.65 * max(positive) + 0.35 * float(np.mean(positive))
            output.append({"support": round(float(np.clip(support, 0, 1)), 6), "neighbors": neighbors})
        return output

    def search(self, event: dict[str, Any], k: int = 5) -> dict[str, Any]:
        return self.search_many([event], k)[0]

    def support(self, event: dict[str, Any], k: int = 5) -> float:
        return float(self.search(event, k)["support"])
