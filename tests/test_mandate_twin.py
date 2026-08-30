import copy
from datetime import datetime, timezone

import pytest

from defend.features.structural import structural_features
from sim.mandate_twin import (
    DETERMINISTIC_SIGNING_AVAILABLE,
    KeyRegistry,
    build_chain,
    resign_intent,
    verify_chain,
)


def _chain(registry: KeyRegistry) -> dict:
    return build_chain(
        registry,
        user_id="cus_synthetic_1",
        merchant_id="mer_synthetic_1",
        agent_id="agent:shopper-v3",
        category="running_shoes",
        budget_minor=500_000,
        mcc="5661",
        prompt_playback="Find running shoes under INR 5000",
        signed_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
        items=[{"sku": "SHOE-77", "name": "Trail Runner", "price_minor": 478_000, "qty": 1}],
        payee_did="did:example:merchant-synthetic-1",
        presence_flag="NOT_PRESENT",
        delegation_depth=2,
    )


def test_legitimate_p256_chain_and_der_signatures_verify() -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    trusted_registry = registry.public_key_fingerprints()

    assert registry.private_key("user:cus_synthetic_1").curve.name == "secp256r1"
    assert bytes.fromhex(chain["intent"]["user_signature"])[0] == 0x30
    assert bytes.fromhex(chain["cart"]["merchant_signature"])[0] == 0x30
    assert bytes.fromhex(chain["payment"]["agent_signature"])[0] == 0x30
    assert all(verify_chain(chain, trusted_registry).values())
    assert all("registry_attestation" not in part for part in chain.values())


def test_event_supplied_attacker_key_is_not_a_registry_trust_anchor() -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    trusted_registry = registry.public_key_fingerprints()
    attacker_registry = KeyRegistry(8)
    resign_intent(attacker_registry, chain["intent"])
    chain["key_registry"] = attacker_registry.public_key_fingerprints()

    checks = verify_chain(chain, trusted_registry)
    assert checks["intent_signature_valid"]
    assert not checks["intent_registration_valid"]
    assert structural_features({"mandate_chain": chain}, trusted_registry)["structural_floor"] == 1


def test_missing_signer_key_or_trusted_registry_is_critical() -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    trusted_registry = registry.public_key_fingerprints()
    chain["intent"].pop("signer_public_key")

    checks = verify_chain(chain, trusted_registry)
    assert not checks["intent_registration_valid"]
    assert structural_features({"mandate_chain": chain}, trusted_registry)["structural_floor"] == 1

    intact = _chain(registry)
    assert not verify_chain(intact)["intent_registration_valid"]
    assert structural_features({"mandate_chain": intact})["structural_floor"] == 1


def test_missing_chain_is_critical_only_for_agentic_checkout() -> None:
    agentic = structural_features(
        {"channel": "agentic_checkout", "mandate_chain": None}
    )
    non_agentic = structural_features({"channel": "card_ecom", "mandate_chain": None})

    assert agentic["signature_chain_valid"] == 0
    assert agentic["signer_did_in_registry"] == 0
    assert agentic["structural_floor"] == 1
    assert non_agentic["signature_chain_valid"] == 1
    assert non_agentic["signer_did_in_registry"] == 1
    assert non_agentic["structural_floor"] == 0


@pytest.mark.parametrize(
    "field",
    (
        "category",
        "max_amount_minor",
        "mcc",
        "payee_allowlist",
        "prompt_playback",
        "signed_at",
        "ttl_seconds",
    ),
)
def test_missing_signed_constraint_field_is_critical(field: str) -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    chain["intent"]["constraint_obj"].pop(field)
    resign_intent(registry, chain["intent"])

    trusted_registry = registry.public_key_fingerprints()
    checks = verify_chain(chain, trusted_registry)
    assert not checks["signed_constraints_complete"]
    assert structural_features({"mandate_chain": chain}, trusted_registry)["structural_floor"] == 1


@pytest.mark.parametrize("chain", ({}, [], False))
def test_malformed_chain_is_critical(chain: object) -> None:
    assert structural_features({"mandate_chain": chain})["structural_floor"] == 1


def test_cart_tamper_breaks_signature_and_hash() -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    tampered = copy.deepcopy(chain)
    tampered["cart"]["items"][0]["price_minor"] += 1

    checks = verify_chain(tampered, registry.public_key_fingerprints())
    assert not checks["cart_signature_valid"]
    assert not checks["hash_matches_cart"]
    assert checks["intent_signature_valid"]
    assert checks["payment_signature_valid"]


def test_playback_tamper_breaks_signed_intent_binding() -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    trusted_registry = registry.public_key_fingerprints()
    tampered = copy.deepcopy(chain)
    tampered["intent"]["prompt_playback"] = "Routine approved purchase summary"

    checks = verify_chain(tampered, trusted_registry)
    assert checks["intent_signature_valid"]
    assert not checks["playback_matches_signed_constraint"]
    assert structural_features({"mandate_chain": chain}, trusted_registry)["structural_floor"] == 0
    assert structural_features({"mandate_chain": tampered}, trusted_registry)["structural_floor"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ttl_seconds", 604_800),
        ("signed_at", "2026-08-28T00:00:00Z"),
        ("category", "crypto_wallet"),
        ("budget_minor", 5_000_000),
    ),
)
def test_unsigned_intent_metadata_cannot_override_signed_constraints(
    field: str, value: object
) -> None:
    registry = KeyRegistry(7)
    chain = _chain(registry)
    tampered = copy.deepcopy(chain)
    tampered["intent"][field] = value

    trusted_registry = registry.public_key_fingerprints()
    checks = verify_chain(tampered, trusted_registry)
    assert checks["intent_signature_valid"]
    assert not checks["intent_metadata_matches_signed_constraint"]
    assert structural_features({"mandate_chain": tampered}, trusted_registry)["structural_floor"] == 1


@pytest.mark.skipif(not DETERMINISTIC_SIGNING_AVAILABLE, reason="cryptography >= 43 required")
def test_seeded_signatures_are_byte_stable() -> None:
    assert _chain(KeyRegistry(99)) == _chain(KeyRegistry(99))
