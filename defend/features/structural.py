"""Deterministic mandate-chain checks."""

from __future__ import annotations

from typing import Any, Mapping

from sim.mandate_twin import verify_chain


def structural_features(
    event: dict[str, Any],
    trusted_registry: Mapping[str, str] | None = None,
) -> dict[str, float]:
    chain = event.get("mandate_chain")
    empty = {
            "hash_matches_cart": 1.0,
            "signature_chain_valid": 1.0,
            "signer_did_in_registry": 1.0,
            "delegation_depth": 0.0,
            "present_flag_vs_behavioral_evidence": 1.0,
            "cart_mcc_in_intent": 1.0,
            "payee_in_allowlist": 1.0,
            "payment_direction_consistency": 1.0,
            "structural_floor": 0.0,
    }
    if chain is None:
        if event.get("channel") == "agentic_checkout":
            return {
                **empty,
                "hash_matches_cart": 0.0,
                "signature_chain_valid": 0.0,
                "signer_did_in_registry": 0.0,
                "structural_floor": 1.0,
            }
        return empty
    if not isinstance(chain, dict) or not chain:
        return {
            **empty,
            "hash_matches_cart": 0.0,
            "signature_chain_valid": 0.0,
            "signer_did_in_registry": 0.0,
            "structural_floor": 1.0,
        }

    intent = chain.get("intent") or {}
    cart = chain.get("cart") or {}
    payment = chain.get("payment") or {}
    try:
        checks = verify_chain(chain, trusted_registry)
        signatures_valid = all(
            checks.get(key, False)
            for key in ("intent_signature_valid", "cart_signature_valid", "payment_signature_valid")
        )
        hash_valid = bool(checks.get("hash_matches_cart") and checks.get("amount_matches_cart"))
        playback_valid = bool(checks.get("playback_matches_signed_constraint", False))
        intent_metadata_valid = bool(
            checks.get("intent_metadata_matches_signed_constraint", False)
        )
        signed_constraints_complete = bool(
            checks.get("signed_constraints_complete", False)
        )
        registration_valid = all(
            checks.get(key, False)
            for key in (
                "intent_registration_valid",
                "cart_registration_valid",
                "payment_registration_valid",
            )
        )
    except (KeyError, TypeError, ValueError):
        signatures_valid = False
        hash_valid = False
        playback_valid = False
        intent_metadata_valid = False
        signed_constraints_complete = False
        registration_valid = False

    constraints = intent.get("constraint_obj") or {}
    allowed_mcc = {str(value) for value in constraints.get("mcc") or []}
    cart_mcc = str(cart.get("mcc") or (event.get("txn") or {}).get("mcc") or "")
    mcc_valid = not allowed_mcc or cart_mcc in allowed_mcc
    allowlist = {str(value) for value in constraints.get("payee_allowlist") or []}
    payee_valid = not allowlist or str(cart.get("payee_did") or "") in allowlist
    txn = event.get("txn") or {}
    registered = registration_valid and bool(txn.get("signer_did_in_registry", True)) and bool(
        payment.get("agent_registered", True)
    )
    depth = max(0, int(payment.get("delegation_depth") or 0))
    trace = event.get("agent_trace") or {}
    observed_presence = trace.get("human_present")
    claimed_presence = str(payment.get("presence_flag") or "") == "PRESENT"
    presence_valid = observed_presence is None or bool(observed_presence) == claimed_presence
    direction_valid = not (
        txn.get("payment_direction") == "COLLECT_DEBIT"
        and txn.get("presented_as") == "CHECKOUT_CREDIT"
    )

    critical_failure = not (
        signatures_valid
        and hash_valid
        and mcc_valid
        and payee_valid
        and playback_valid
        and intent_metadata_valid
        and signed_constraints_complete
        and registration_valid
    )
    secondary_failure = not (registered and presence_valid and direction_valid) or depth > 3
    floor = 1.0 if critical_failure else 0.86 if secondary_failure else 0.0
    return {
        "hash_matches_cart": float(hash_valid),
        "signature_chain_valid": float(signatures_valid),
        "signer_did_in_registry": float(registered),
        "delegation_depth": float(depth),
        "present_flag_vs_behavioral_evidence": float(presence_valid),
        "cart_mcc_in_intent": float(mcc_valid),
        "payee_in_allowlist": float(payee_valid),
        "payment_direction_consistency": float(direction_valid),
        "structural_floor": floor,
    }
