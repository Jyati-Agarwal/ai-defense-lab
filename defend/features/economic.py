"""Market- and mandate-relative economic features."""

from __future__ import annotations

from typing import Any


def economic_features(event: dict[str, Any], market_median: float) -> dict[str, float]:
    txn = event.get("txn") or {}
    chain = event.get("mandate_chain") or {}
    intent = chain.get("intent") or {}
    payment = chain.get("payment") or {}
    amount = float(txn.get("amount_minor") or payment.get("amount_minor") or 0)
    budget = float(intent.get("budget_minor") or (intent.get("constraint_obj") or {}).get("max_amount_minor") or 0)
    fanout = float(txn.get("fanout_count_per_intent") or (event.get("agent_trace") or {}).get("fanout_count") or 1)
    return {
        "price_vs_market_median": amount / max(market_median, 1.0),
        "budget_utilisation_pct": 0.0 if budget <= 0 else amount / budget,
        "fanout_count_per_intent": max(1.0, fanout),
        "payee_novelty": float(bool(txn.get("is_new_beneficiary"))),
    }
