"""Blue policy and intentionally old-fashioned comparison rules."""

from __future__ import annotations

from typing import Any


def action_for_score(score: float) -> str:
    if score < 0.30:
        return "ALLOW"
    if score < 0.60:
        return "STEP_UP"
    if score < 0.85:
        return "HOLD"
    return "BLOCK"


def legacy_rules_action(event: dict[str, Any], amount_limit_minor: int = 5_000_000) -> str:
    txn = event.get("txn") or {}
    chain = event.get("mandate_chain") or {}
    payment = chain.get("payment") or {}
    signature_invalid = any(
        not bool((chain.get(name) or {}).get("signature_valid", True)) for name in ("intent", "cart", "payment")
    )
    if signature_invalid or bool(txn.get("blocklisted_bin")):
        return "BLOCK"
    if float(txn.get("geo_velocity_kmh") or 0) > 900:
        return "HOLD"
    if float(txn.get("amount_minor") or payment.get("amount_minor") or 0) > amount_limit_minor:
        return "STEP_UP"
    return "ALLOW"


def policy_decision(event: dict[str, Any], score: float, reason_codes: list[str], latency_ms: float) -> dict[str, Any]:
    return {
        "action": action_for_score(score),
        "legacy_rules_action": legacy_rules_action(event),
        "reason_codes": list(reason_codes),
        "latency_ms": round(float(latency_ms), 6),
    }
