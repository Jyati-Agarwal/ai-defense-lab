"""MiniLM-backed intent-to-cart consistency features."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from defend.memory.precedent import intent_text, purchase_text


TOKENS = re.compile(r"[a-z0-9]+")


def semantic_texts(event: dict[str, Any]) -> tuple[str, str]:
    return intent_text(event), purchase_text(event)


def semantic_features(
    event: dict[str, Any],
    intent_vector: np.ndarray,
    purchase_vector: np.ndarray,
    precedent_support: float,
) -> dict[str, float]:
    intent_value, purchase_value = semantic_texts(event)
    if not intent_value or not purchase_value:
        return {
            "intent_cart_semantic_drift": 0.0,
            "precedent_support": 0.0,
            "contextualized_semantic_drift": 0.0,
            "playback_constraint_consistency": 1.0,
            "category_match": 1.0,
            "intent_specificity": 0.0,
        }
    cosine = float(np.clip(np.dot(intent_vector, purchase_vector), -1, 1))
    drift = float(np.clip(1 - cosine, 0, 1))
    chain = event.get("mandate_chain") or {}
    intent = chain.get("intent") or {}
    cart = chain.get("cart") or {}
    payment = chain.get("payment") or {}
    constraints = intent.get("constraint_obj") or {}
    maximum = float(constraints.get("max_amount_minor") or intent.get("budget_minor") or 0)
    amount = float(payment.get("amount_minor") or cart.get("total_minor") or 0)
    amount_ok = maximum <= 0 or amount <= maximum
    allowed_mcc = {str(value) for value in constraints.get("mcc") or []}
    observed_mcc = str(cart.get("mcc") or (event.get("txn") or {}).get("mcc") or "")
    mcc_ok = not allowed_mcc or observed_mcc in allowed_mcc
    constraints_ok = amount_ok and mcc_ok
    category = str(intent.get("category") or "").replace("_", " ").lower()
    category_tokens = set(TOKENS.findall(category))
    playback_tokens = set(TOKENS.findall(str(intent.get("prompt_playback") or "").lower()))
    playback_ok = not category_tokens or bool(category_tokens & playback_tokens)
    purchase_tokens = set(TOKENS.findall(purchase_value.lower()))
    category_match = not category_tokens or bool(category_tokens & purchase_tokens)
    prompt_tokens = TOKENS.findall(intent_value.lower())
    specificity = min(1.0, len(set(prompt_tokens)) / 12 + len(constraints) / 8)
    return {
        "intent_cart_semantic_drift": drift,
        "precedent_support": float(np.clip(precedent_support, 0, 1)),
        "contextualized_semantic_drift": drift * (1 - float(np.clip(precedent_support, 0, 1))),
        "playback_constraint_consistency": float(constraints_ok and playback_ok),
        "category_match": float(category_match),
        "intent_specificity": specificity,
    }
