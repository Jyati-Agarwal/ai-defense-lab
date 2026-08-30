"""Composable mutations spanning the payment lifecycle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sim.content_llm import ContentCache
from sim.mandate_twin import (
    KeyRegistry,
    annotate_chain,
    relink_payment,
    resign_cart,
    resign_payment,
)


LIFECYCLE = (
    "reconnaissance",
    "context_manip",
    "intent_manip",
    "tool_manip",
    "merchant_manip",
    "payment_manip",
    "delayed_exec",
    "cash_out",
)


@dataclass(frozen=True)
class AttackDefinition:
    atlas_id: str
    name: str
    family: str
    split: str
    primitives: tuple[str, ...]
    intended_signals: tuple[str, ...]
    base_channel: str = "agentic_checkout"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = [LIFECYCLE.index(primitive) for primitive in self.primitives]
        if positions != sorted(set(positions)):
            raise ValueError(f"{self.name}: primitives must be unique and lifecycle ordered")
        if self.split not in {"A", "B", "C", "D"}:
            raise ValueError(f"{self.name}: fraudulent split must be A, B, C, or D")


def _trace(event: dict[str, Any]) -> dict[str, Any]:
    if event["agent_trace"] is None:
        event["agent_trace"] = {"tool_calls": [], "pages_read": [], "negotiation_seconds": 0.0}
    return event["agent_trace"]


def _require_registry(registry: KeyRegistry | None) -> KeyRegistry:
    if registry is None:
        raise ValueError("a KeyRegistry is required to produce cryptographically valid attack fixtures")
    return registry


def reconnaissance(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    event["txn"]["recon_attempts_10m"] = int(params.get("recon_attempts", 8))
    if event["agent_trace"] is not None:
        _trace(event)["tool_calls"].insert(
            0,
            {"t": -1, "tool": "merchant_discovery", "topic": params.get("recon_topic", "price and payee")},
        )


def context_manip(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    language = params.get("language", "en")
    artifact = ContentCache().require("injection", language)
    metadata = artifact["artifact_metadata"]
    _trace(event)["pages_read"].append(
        {
            "url": "https://safe-artifact.invalid/non-operational",
            "hidden_text_present": metadata["hidden_text_present"],
            "hidden_text_sample": artifact["content"],
            "zero_width_chars": metadata["zero_width_chars"],
            "invisible_css": metadata["invisible_css"],
            "artifact_id": artifact["artifact_id"],
            "language": language,
        }
    )
    event["txn"]["content_provenance_anomaly"] = True


def intent_manip(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    chain = event["mandate_chain"]
    if chain:
        if params.get("playback_mismatch"):
            chain["intent"]["prompt_playback"] = "Routine approved purchase summary [SAFE SYNTHETIC]"
            event["txn"]["playback_constraint_mismatch"] = True
        event["txn"]["intent_topic_shift"] = params.get("shift_topic", "unrelated payment topic")
    else:
        event["txn"]["social_pretext_present"] = True


def tool_manip(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    trace = _trace(event)
    next_t = max((call.get("t", 0) for call in trace["tool_calls"]), default=0) + 1
    trace["tool_calls"].append(
        {"t": next_t, "tool": params.get("tool", "create_payment_mandate"), "topic": params.get("tool_topic", "unrelated payment topic")}
    )
    event["txn"]["tool_call_topic_shift"] = float(params.get("topic_shift", 0.94))
    chain = event["mandate_chain"]
    if chain and params.get("delegation_hijack"):
        payment = chain["payment"]
        payment["delegation_depth"] = int(params.get("delegation_depth", 7))
        payment["agent_registered"] = False
        event["txn"]["delegation_scope_violation"] = True
        resign_payment(_require_registry(registry), payment)


def merchant_manip(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    mode = params.get("merchant_mode")
    chain = event["mandate_chain"]
    if mode == "fanout":
        fanout = int(params.get("fanout", 4))
        event["txn"]["intent_fanout_count"] = fanout
        event["txn"]["fanout_count_per_intent"] = fanout
        _trace(event)["fanout_count"] = fanout
        event["txn"]["fanout_payees"] = [f"did:example:fanout-safe-{number}" for number in range(1, 5)]
        return
    if mode == "upi_collect":
        event["txn"]["requester_vpa_novel"] = True
        return
    if chain is None:
        return
    key_registry = _require_registry(registry)
    cart = chain["cart"]
    if mode == "substitute_category":
        budget = chain["intent"]["budget_minor"]
        amount = min(budget, max(cart["total_minor"], round(cart["total_minor"] * 1.4)))
        cart["items"] = [{"sku": "SAFE-DEMO-UNRELATED", "name": "Digital wallet voucher", "price_minor": amount, "qty": 1}]
        cart["total_minor"] = amount
        event["txn"].update(amount_minor=amount, intent_cart_semantic_drift=0.98)
        resign_cart(key_registry, cart)
        relink_payment(key_registry, chain)
    elif mode == "price_inflate":
        market_median = int(event["txn"].get("market_median_minor") or cart["total_minor"])
        intent = chain["intent"]
        budget = int(intent["constraint_obj"]["max_amount_minor"])
        original_total = int(cart["total_minor"])
        target = min(
            budget,
            max(original_total + 1, round(market_median * float(params.get("price_ratio", 1.85)))),
        )
        cart["items"][0]["price_minor"] = target
        cart["items"][0]["qty"] = 1
        cart["total_minor"] = target
        event["txn"].update(
            amount_minor=target,
            price_vs_market_median=target / max(market_median, 1),
            price_inflation_ratio=target / max(original_total, 1),
            budget_utilisation_pct=target / budget,
        )
        resign_cart(key_registry, cart)
        relink_payment(key_registry, chain)
    elif mode == "fake_did":
        cart["payee_did"] = "did:example:unregistered-safe-demo"
        event["txn"].update(signer_did_in_registry=False, payee_allowlist_match=False)
        relink_payment(key_registry, chain)


def payment_manip(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    mode = params.get("payment_mode")
    chain = event["mandate_chain"]
    if mode == "card_testing":
        event["txn"].update(
            amount_minor=100,
            cards_tested_10m=int(params.get("cards_tested", 19)),
            authorisation_attempts_10m=int(params.get("attempts", 27)),
            decline_ratio_10m=0.89,
        )
        return
    if mode == "upi_collect":
        event["channel"] = "upi_collect"
        event["rail"] = "upi"
        event["txn"].update(payment_direction="COLLECT_DEBIT", presented_as="CHECKOUT_CREDIT", is_new_beneficiary=True)
        return
    if chain is None:
        return
    key_registry = _require_registry(registry)
    payment = chain["payment"]
    if mode == "hash_decouple":
        payment["cart_hash"] = "sha256:" + "0" * 64
        resign_payment(key_registry, payment)
    elif mode == "presence_lie":
        payment["presence_flag"] = "PRESENT"
        _trace(event)["human_present"] = False
        event["txn"]["human_interaction_events"] = 0
        event["txn"]["presence_evidence_mismatch"] = True
        resign_payment(key_registry, payment)


def delayed_exec(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    chain = event["mandate_chain"]
    if chain is None:
        event["txn"]["dormancy_days"] = int(params.get("dormancy_days", 120))
        return
    intent = chain["intent"]
    signed_at = datetime.fromisoformat(intent["signed_at"].replace("Z", "+00:00"))
    delay_seconds = int(params.get("delay_seconds", intent["ttl_seconds"] + 3_600))
    event["ts"] = (signed_at + timedelta(seconds=delay_seconds)).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    event["txn"]["hour_local"] = datetime.fromisoformat(event["ts"].replace("Z", "+00:00")).hour
    event["txn"]["execution_delay_seconds"] = delay_seconds
    event["txn"]["ttl_utilisation_pct"] = delay_seconds / intent["ttl_seconds"]


def cash_out(event: dict[str, Any], params: dict[str, Any], registry: KeyRegistry | None) -> None:
    event["txn"].update(
        account_age_days=int(params.get("account_age_days", 420)),
        prior_utilisation_pct=float(params.get("prior_utilisation_pct", 0.08)),
        current_utilisation_pct=float(params.get("current_utilisation_pct", 0.98)),
        cashout_velocity_1h=int(params.get("cashout_velocity", 11)),
        beneficiary_in_degree=max(int(event["txn"].get("beneficiary_in_degree", 0)), int(params.get("beneficiary_in_degree", 37))),
    )


PRIMITIVES: dict[str, Callable[[dict[str, Any], dict[str, Any], KeyRegistry | None], None]] = {
    "reconnaissance": reconnaissance,
    "context_manip": context_manip,
    "intent_manip": intent_manip,
    "tool_manip": tool_manip,
    "merchant_manip": merchant_manip,
    "payment_manip": payment_manip,
    "delayed_exec": delayed_exec,
    "cash_out": cash_out,
}


def apply_definition(
    event: dict[str, Any],
    attack: AttackDefinition,
    *,
    registry: KeyRegistry | None = None,
) -> dict[str, Any]:
    mutated = copy.deepcopy(event)
    for primitive in attack.primitives:
        PRIMITIVES[primitive](mutated, attack.parameters, registry)
    if mutated["mandate_chain"]:
        annotate_chain(mutated["mandate_chain"])
    mutated["label"] = {
        "is_fraud": True,
        "atlas_id": attack.atlas_id,
        "attack_primitives": list(attack.primitives),
        "attack_family": attack.family,
        "arena_round": 1 if attack.split == "B" else None,
        "split": attack.split,
    }
    return mutated


def has_signal(event: dict[str, Any], signal: str) -> bool:
    txn = event["txn"]
    chain = event["mandate_chain"]
    checks = {
        "hidden_content_provenance": lambda: any(page.get("hidden_text_present") for page in _trace(event)["pages_read"]),
        "intent_cart_semantic_drift": lambda: txn.get("intent_cart_semantic_drift", 0) >= 0.8,
        "price_vs_market_median": lambda: txn.get("price_inflation_ratio", 0) > 1,
        "playback_constraint_mismatch": lambda: txn.get("playback_constraint_mismatch") is True,
        "ttl_expired": lambda: txn.get("ttl_utilisation_pct", 0) > 1,
        "presence_evidence_mismatch": lambda: txn.get("presence_evidence_mismatch") is True,
        "intent_fanout": lambda: txn.get("intent_fanout_count", 0) > 1,
        "cart_hash_mismatch": lambda: bool(chain) and chain["payment"].get("hash_matches_cart") is False,
        "delegation_scope_violation": lambda: txn.get("delegation_scope_violation") is True,
        "merchant_registry_miss": lambda: txn.get("signer_did_in_registry") is False,
        "card_testing_velocity": lambda: txn.get("cards_tested_10m", 0) >= 10,
        "bust_out_regime_change": lambda: txn.get("current_utilisation_pct", 0) - txn.get("prior_utilisation_pct", 0) > 0.7,
        "collect_direction_mismatch": lambda: txn.get("payment_direction") == "COLLECT_DEBIT" and txn.get("presented_as") == "CHECKOUT_CREDIT",
    }
    if signal not in checks:
        raise KeyError(f"unknown intended signal: {signal}")
    return bool(checks[signal]())
