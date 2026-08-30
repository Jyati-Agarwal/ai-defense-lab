"""Single frozen-model inference entry point used by the interactive composer."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import joblib

from defend.embeddings import CachedMiniLM
from defend.features import FeatureExtractor
from defend.memory import PrecedentMemory
from defend.models import DefenseModel
from defend.models.bundle import LAYER_ORDER
from defend.policy import policy_decision
from sim.mandate_twin import (
    KeyRegistry,
    SYNTHETIC_KEY_SCOPE,
    build_cart,
    build_intent,
    build_payment,
    key_registry_sha256,
    validate_synthetic_key_registry,
)


@lru_cache(maxsize=4)
def _runtime(project_root: str) -> tuple[DefenseModel, FeatureExtractor]:
    root = Path(project_root)
    results = root / "results"
    model_path = results / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError("results/model.pkl is missing; run the seeded pipeline first")
    model: DefenseModel = joblib.load(model_path)
    embedder = CachedMiniLM(results / "embeddings.npz")
    memory = PrecedentMemory(embedder, results / "precedent.index", results / "precedent_history.json")
    manifest = json.loads((root / "data" / "events.manifest.json").read_text(encoding="utf-8"))
    trusted_registry = validate_synthetic_key_registry(
        manifest.get("key_registry"), manifest.get("synthetic_identity_seed")
    )
    if (
        not trusted_registry
        or manifest.get("key_registry_scope") != SYNTHETIC_KEY_SCOPE
        or manifest.get("key_registry_sha256") != key_registry_sha256(trusted_registry)
    ):
        raise RuntimeError("release-bound event key registry is invalid")
    return model, FeatureExtractor(
        embedder,
        memory,
        model.history,
        trusted_registry=trusted_registry,
    )


def _composer_event(
    payload: dict[str, Any],
    trusted_registry: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    event_id = "evt_composer_" + hashlib.sha256(canonical.encode()).hexdigest()[:12]
    timestamp = datetime(2026, 8, 27, 14, 30, tzinfo=timezone.utc)
    channel = str(payload.get("channel") or "agentic_checkout")
    amount = max(1, int(payload.get("amount_minor") or 1))
    txn = {
        "amount_minor": amount,
        "currency": "INR",
        "mcc": "6051" if payload.get("injection_style") != "none" else "5661",
        "customer_id": "cus_composer_synthetic",
        "device_id": "dev_composer_synthetic",
        "geo": {"lat": 19.076, "lon": 72.8777},
        "hour_local": timestamp.hour,
        "beneficiary_id": "ben_composer_synthetic",
        "is_new_beneficiary": True,
        "beneficiary_in_degree": 1,
        "fanout_count_per_intent": int(payload.get("fanout") or 1),
        "shared_device_cluster_size": max(1, int(payload.get("hops") or 0)),
        "connected_component_size": max(1, int(payload.get("hops") or 0) + int(payload.get("fanout") or 1)),
    }
    if channel != "agentic_checkout":
        return {
            "event_id": event_id,
            "ts": timestamp.isoformat().replace("+00:00", "Z"),
            "channel": channel,
            "rail": "card" if channel == "card_ecom" else "upi" if channel.startswith("upi") else "wire",
            "mandate_chain": None,
            "agent_trace": None,
            "txn": txn,
        }

    principals = trusted_registry or {}
    user_id = next(
        (principal.removeprefix("user:") for principal in principals if principal.startswith("user:")),
        "cus_composer_synthetic",
    )
    merchant_id = next(
        (
            principal.removeprefix("merchant:")
            for principal in principals
            if principal.startswith("merchant:")
        ),
        "merchant-composer",
    )
    agent_id = next(
        (principal.removeprefix("agent:") for principal in principals if principal.startswith("agent:")),
        "agent:composer-synthetic",
    )
    txn.update(customer_id=user_id, merchant_id=merchant_id)
    registry = KeyRegistry()
    injected = str(payload.get("injection_style") or "none") != "none"
    intent = build_intent(
        registry,
        user_id=user_id,
        category="running_shoes",
        budget_minor=max(amount, 500_000),
        mcc=["5661"],
        prompt_playback="Find running shoes under INR 5000",
        signed_at=timestamp - timedelta(seconds=int(payload.get("ttl_delay_seconds") or 60)),
        payee_allowlist=["did:example:merchant-shoes"] if not injected else [],
    )
    cart = build_cart(
        registry,
        merchant_id=merchant_id,
        items=[
            {
                "sku": "SYNTHETIC-001",
                "name": "Digital wallet voucher" if injected else "Trail running shoes",
                "price_minor": amount,
                "qty": 1,
            }
        ],
        payee_did="did:example:merchant-wallet" if injected else "did:example:merchant-shoes",
        mcc="6051" if injected else "5661",
    )
    payment = build_payment(
        registry,
        cart=cart,
        agent_id=agent_id,
        presence_flag="PRESENT" if payload.get("lie_about_presence") else "NOT_PRESENT",
        delegation_depth=int(payload.get("delegation_depth") or 0),
    )
    style = str(payload.get("injection_style") or "none")
    page = {
        "url": "https://synthetic-merchant.example/product",
        "hidden_text_present": style == "hidden_text",
        "hidden_text_sample": "[TRUNCATED / WATERMARKED / NON-OPERATIONAL INJECTION]" if injected else "",
        "zero_width_chars": 14 if style == "zero_width" else 0,
        "invisible_css": style == "hidden_text",
    }
    return {
        "event_id": event_id,
        "ts": timestamp.isoformat().replace("+00:00", "Z"),
        "channel": channel,
        "rail": "card",
        "mandate_chain": {"intent": intent, "cart": cart, "payment": payment},
        "agent_trace": {
            "tool_calls": [
                {"t": 0, "tool": "web_search", "topic": "running shoes"},
                {"t": 1, "tool": "fetch_page", "topic": "wallet" if injected else "running shoes"},
                {"t": 2, "tool": "create_payment_mandate", "topic": "wallet" if injected else "running shoes"},
            ],
            "pages_read": [page],
            "negotiation_seconds": max(0.2, 15 - 12 * float(payload.get("evasion_aggressiveness") or 0)),
            "human_present": False,
        },
        "txn": txn,
    }


def score_composer(payload: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    started = perf_counter()
    model, extractor = _runtime(str(Path(project_root).resolve()))
    event = _composer_event(payload, extractor.trusted_registry)
    prediction = model.predict(extractor.transform([event]))
    layers = prediction["layers"][0]
    fused = float(prediction["fused"][0])
    event["scores"] = {name: round(float(layers[index]), 8) for index, name in enumerate(LAYER_ORDER)}
    event["scores"]["fused"] = round(fused, 8)
    event["decision"] = policy_decision(
        event,
        fused,
        list(prediction["reason_codes"][0]),
        (perf_counter() - started) * 1_000,
    )
    return event
