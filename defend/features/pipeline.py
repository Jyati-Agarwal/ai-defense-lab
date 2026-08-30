"""Batch feature extraction with past-only entity state."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from defend.embeddings import CachedMiniLM
from defend.memory import PrecedentMemory
from sim.mandate_twin import validate_trusted_registry

from .economic import economic_features
from .graph import graph_features
from .provenance import domains, provenance_features, topics
from .semantic import semantic_features, semantic_texts
from .structural import structural_features
from .temporal import parse_time, temporal_features


FEATURE_GROUPS = {
    "structural": (
        "hash_matches_cart",
        "signature_chain_valid",
        "signer_did_in_registry",
        "delegation_depth",
        "present_flag_vs_behavioral_evidence",
        "cart_mcc_in_intent",
        "payee_in_allowlist",
        "payment_direction_consistency",
        "structural_floor",
    ),
    "semantic": (
        "intent_cart_semantic_drift",
        "precedent_support",
        "contextualized_semantic_drift",
        "playback_constraint_consistency",
        "category_match",
        "intent_specificity",
    ),
    "provenance": (
        "injection_fingerprint_score",
        "tool_call_topic_shift",
        "retrieved_domain_novelty",
        "negotiation_speed_zscore",
    ),
    "temporal": (
        "ttl_utilisation_pct",
        "hour_deviation",
        "amount_zscore_entity",
        "velocity_1h",
        "velocity_24h",
        "credit_utilisation_regime_change",
    ),
    "economic": (
        "price_vs_market_median",
        "budget_utilisation_pct",
        "fanout_count_per_intent",
        "payee_novelty",
        "beneficiary_in_degree",
        "shared_device_cluster_size",
        "connected_component_size",
    ),
}
FEATURE_COLUMNS = tuple(value for values in FEATURE_GROUPS.values() for value in values)
LAYER_COLUMNS = tuple(f"layer_{name}" for name in FEATURE_GROUPS)


@dataclass
class FeatureBatch:
    frame: pd.DataFrame
    text_vectors: np.ndarray
    has_text: np.ndarray
    precedents: list[dict[str, Any]]


def _mean(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default


class FeatureExtractor:
    def __init__(
        self,
        embedder: CachedMiniLM,
        memory: PrecedentMemory | None = None,
        history: Iterable[dict[str, Any]] = (),
        market_path: str | Path | None = None,
        trusted_registry: Mapping[str, str] | None = None,
    ) -> None:
        self.embedder = embedder
        self.memory = memory
        self.history = list(history)
        self.market = self._market_medians(market_path)
        self.trusted_registry = (
            validate_trusted_registry(trusted_registry) if trusted_registry is not None else {}
        )

    @staticmethod
    def _market_medians(path: str | Path | None) -> dict[str, float]:
        candidate = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "market_prices.csv"
        if not candidate.is_file():
            return {}
        with candidate.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row["mcc"])].append(float(row["median_minor"]))
        return {key: _mean(values, 1.0) for key, values in grouped.items()}

    def transform(self, events: Iterable[dict[str, Any]]) -> FeatureBatch:
        rows = list(events)
        if not rows:
            return FeatureBatch(
                pd.DataFrame(columns=FEATURE_COLUMNS + LAYER_COLUMNS, dtype=float),
                np.empty((0, 384), dtype=np.float32),
                np.empty(0, dtype=bool),
                [],
            )
        semantic_pairs = [semantic_texts(event) for event in rows]
        intent_vectors = self.embedder.encode(pair[0] for pair in semantic_pairs)
        purchase_vectors = self.embedder.encode(pair[1] for pair in semantic_pairs)
        has_text = np.asarray([bool(left and right) for left, right in semantic_pairs], dtype=bool)
        text_vectors = np.abs(intent_vectors - purchase_vectors).astype(np.float32, copy=False)
        precedent_rows = (
            self.memory.search_many(rows) if self.memory is not None else [{"support": 0.0, "neighbors": []} for _ in rows]
        )

        all_topics = [topics(event) for event in rows]
        flat_topics = [value for group in all_topics for value in group]
        encoded_topics = self.embedder.encode(flat_topics)
        topic_offsets = []
        cursor = 0
        for group in all_topics:
            topic_offsets.append(encoded_topics[cursor : cursor + len(group)])
            cursor += len(group)

        known_domains = {domain for event in self.history for domain in domains(event)}
        negotiations = [
            float((event.get("agent_trace") or {}).get("negotiation_seconds"))
            for event in self.history
            if (event.get("agent_trace") or {}).get("negotiation_seconds") is not None
        ]
        negotiation_mean = _mean(negotiations, 10.0)
        negotiation_std = float(np.std(negotiations)) if len(negotiations) > 1 else 5.0

        times: dict[str, list[datetime]] = defaultdict(list)
        hours: dict[str, list[float]] = defaultdict(list)
        amounts: dict[str, list[float]] = defaultdict(list)
        for event in sorted(self.history, key=lambda value: str(value.get("ts", ""))):
            txn = event.get("txn") or {}
            customer = str(txn.get("customer_id") or "unknown")
            timestamp = parse_time(str(event["ts"]))
            times[customer].append(timestamp)
            hours[customer].append(float(txn.get("hour_local", timestamp.hour)))
            amounts[customer].append(float(txn.get("amount_minor") or 0))

        output: list[dict[str, float]] = [{} for _ in rows]
        ordered = sorted(enumerate(rows), key=lambda item: str(item[1].get("ts", "")))
        for position, event in ordered:
            txn = event.get("txn") or {}
            customer = str(txn.get("customer_id") or "unknown")
            timestamp = parse_time(str(event["ts"]))
            amount = float(txn.get("amount_minor") or 0)
            values: dict[str, float] = {}
            values.update(structural_features(event, self.trusted_registry))
            values.update(
                semantic_features(
                    event,
                    intent_vectors[position],
                    purchase_vectors[position],
                    float(precedent_rows[position]["support"]),
                )
            )
            values.update(
                provenance_features(
                    event,
                    topic_offsets[position],
                    known_domains,
                    negotiation_mean,
                    negotiation_std,
                )
            )
            values.update(
                temporal_features(
                    event,
                    times[customer],
                    _mean(hours[customer], 18.0),
                    _mean(amounts[customer], max(amount, 1.0)),
                )
            )
            values["structural_floor"] = max(
                values["structural_floor"],
                float(values["ttl_utilisation_pct"] > 1),
                0.60 * values["credit_utilisation_regime_change"],
            )
            market_median = self.market.get(str(txn.get("mcc") or ""), max(amount, 1.0))
            values.update(economic_features(event, market_median))
            values.update(graph_features(event))
            values.update(self._layer_seeds(values))
            output[position] = values
            times[customer].append(timestamp)
            hours[customer].append(float(txn.get("hour_local", timestamp.hour)))
            amounts[customer].append(amount)

        self.embedder.flush()
        frame = pd.DataFrame(output, columns=FEATURE_COLUMNS + LAYER_COLUMNS).fillna(0.0).astype(float)
        return FeatureBatch(frame, text_vectors, has_text, precedent_rows)

    @staticmethod
    def _layer_seeds(values: dict[str, float]) -> dict[str, float]:
        structural = float(values["structural_floor"])
        semantic = np.clip(
            0.62 * values["contextualized_semantic_drift"]
            + 0.20 * (1 - values["playback_constraint_consistency"])
            + 0.18 * (1 - values["category_match"]),
            0,
            1,
        )
        provenance = np.clip(
            0.52 * values["injection_fingerprint_score"]
            + 0.28 * values["tool_call_topic_shift"]
            + 0.12 * values["retrieved_domain_novelty"]
            + 0.08 * min(values["negotiation_speed_zscore"] / 3, 1),
            0,
            1,
        )
        temporal = np.clip(
            0.20 * min(values["ttl_utilisation_pct"], 1)
            + 0.10 * values["hour_deviation"]
            + 0.15 * min(values["amount_zscore_entity"] / 4, 1)
            + 0.40 * min(values["velocity_1h"] / 5, 1)
            + 0.15 * min(values["velocity_24h"] / 12, 1),
            0,
            1,
        )
        economic = np.clip(
            0.18 * max(0, min((values["price_vs_market_median"] - 1) / 3, 1))
            + 0.15 * min(values["budget_utilisation_pct"], 1)
            + 0.18 * min((values["fanout_count_per_intent"] - 1) / 5, 1)
            + 0.14 * values["payee_novelty"]
            + 0.20 * min(values["beneficiary_in_degree"] / 25, 1)
            + 0.08 * min(values["shared_device_cluster_size"] / 10, 1)
            + 0.07 * min(values["connected_component_size"] / 25, 1),
            0,
            1,
        )
        return {
            "layer_structural": structural,
            "layer_semantic": float(semantic),
            "layer_provenance": float(provenance),
            "layer_temporal": float(temporal),
            "layer_economic": float(economic),
        }
