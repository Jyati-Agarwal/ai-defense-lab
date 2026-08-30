"""Cryptographic intent, cart, and payment mandate twin."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


# Order of NIST P-256, used only to derive reproducible fixture keys from a seed.
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
SYNTHETIC_IDENTITY_SEED = "ai-defense-lab:public-synthetic-identities:v1"
SYNTHETIC_KEY_SCOPE = "synthetic fixtures only; public deterministic keys; never production credentials"
SIGNED_CONSTRAINT_FIELDS = frozenset(
    {
        "category",
        "max_amount_minor",
        "mcc",
        "payee_allowlist",
        "prompt_playback",
        "signed_at",
        "ttl_seconds",
    }
)
try:
    ec.ECDSA(hashes.SHA256(), deterministic_signing=True)
    DETERMINISTIC_SIGNING_AVAILABLE = True
except TypeError:
    DETERMINISTIC_SIGNING_AVAILABLE = False


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _signature_algorithm() -> ec.ECDSA:
    if DETERMINISTIC_SIGNING_AVAILABLE:
        return ec.ECDSA(hashes.SHA256(), deterministic_signing=True)
    # cryptography < 43 verifies correctly but cannot make byte-stable fixtures.
    return ec.ECDSA(hashes.SHA256())


def _public_key_hex(key: ec.EllipticCurvePublicKey) -> str:
    return key.public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    ).hex()


def verify_signature(
    payload: Any, signature_hex: str | None, public_key_hex: str | None
) -> bool:
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), canonical_json(payload), ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, TypeError, ValueError):
        return False
    return True


def public_key_fingerprint(public_key_hex: str) -> str:
    encoded = bytes.fromhex(public_key_hex)
    key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), encoded)
    if _public_key_hex(key) != public_key_hex.lower():
        raise ValueError("public key must use canonical uncompressed P-256 encoding")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_trusted_registry(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("trusted key registry must be a mapping")
    registry = dict(value)
    if any(
        not isinstance(principal, str)
        or not principal
        or not isinstance(fingerprint, str)
        or not fingerprint.startswith("sha256:")
        or len(fingerprint) != 71
        or any(character not in "0123456789abcdef" for character in fingerprint[7:])
        for principal, fingerprint in registry.items()
    ):
        raise ValueError("trusted key registry contains an invalid principal or fingerprint")
    return dict(sorted(registry.items()))


@dataclass
class KeyRegistry:
    """Synthetic-only P-256 keys derived from a public seed for reproducible fixtures."""

    seed: int | str = SYNTHETIC_IDENTITY_SEED
    _keys: dict[str, ec.EllipticCurvePrivateKey] = field(default_factory=dict)

    def private_key(self, principal: str) -> ec.EllipticCurvePrivateKey:
        if principal not in self._keys:
            digest = hashlib.sha256(f"{self.seed}:{principal}".encode()).digest()
            scalar = int.from_bytes(digest, "big") % (P256_ORDER - 1) + 1
            self._keys[principal] = ec.derive_private_key(scalar, ec.SECP256R1())
        return self._keys[principal]

    def sign(self, principal: str, payload: Any) -> tuple[str, str]:
        key = self.private_key(principal)
        signature = key.sign(canonical_json(payload), _signature_algorithm())
        return signature.hex(), _public_key_hex(key.public_key())

    def public_key_fingerprints(self) -> dict[str, str]:
        return {
            principal: public_key_fingerprint(_public_key_hex(key.public_key()))
            for principal, key in sorted(self._keys.items())
        }

    def manifest_fields(self) -> dict[str, Any]:
        key_registry = self.public_key_fingerprints()
        return {
            "synthetic_identity_seed": str(self.seed),
            "key_registry_scope": SYNTHETIC_KEY_SCOPE,
            "key_registry": key_registry,
            "key_registry_sha256": key_registry_sha256(key_registry),
        }


def expected_synthetic_key_registry(principals: Iterable[str]) -> dict[str, str]:
    registry = KeyRegistry()
    for principal in principals:
        registry.private_key(principal)
    return registry.public_key_fingerprints()


def key_registry_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(validate_trusted_registry(value))).hexdigest()


def observed_event_key_registry(events: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for event in events:
        chain = event.get("mandate_chain")
        if chain is None:
            continue
        if not isinstance(chain, Mapping):
            raise ValueError("mandate_chain must be an object or null")
        for section, identifier, prefix in (
            ("intent", "user_id", "user:"),
            ("cart", "merchant_id", "merchant:"),
            ("payment", "agent_id", "agent:"),
        ):
            part = chain.get(section)
            if not isinstance(part, Mapping):
                raise ValueError(f"mandate_chain.{section} must be an object")
            identity = part.get(identifier)
            public_key = part.get("signer_public_key")
            if not isinstance(identity, str) or not identity or not isinstance(public_key, str):
                raise ValueError(f"mandate_chain.{section} signer identity or key is missing")
            principal = prefix + identity
            fingerprint = public_key_fingerprint(public_key)
            if principal in observed and observed[principal] != fingerprint:
                raise ValueError(f"principal {principal!r} uses multiple public keys")
            observed[principal] = fingerprint
    return dict(sorted(observed.items()))


def validate_synthetic_key_registry(
    value: Any,
    identity_seed: Any,
) -> dict[str, str]:
    if identity_seed != SYNTHETIC_IDENTITY_SEED:
        raise ValueError("event manifest does not declare the fixed synthetic identity seed")
    declared = validate_trusted_registry(value)
    if declared != expected_synthetic_key_registry(declared):
        raise ValueError("event manifest key registry does not match the fixed synthetic identity seed")
    return declared


def validate_synthetic_event_registry(
    events: Iterable[Mapping[str, Any]],
    value: Any,
    identity_seed: Any,
) -> dict[str, str]:
    declared = validate_synthetic_key_registry(value, identity_seed)
    if declared != observed_event_key_registry(events):
        raise ValueError("event manifest key registry does not exactly cover observed signers")
    return declared


def _registration_valid(
    principal: str,
    public_key_hex: str | None,
    trusted_registry: Mapping[str, str] | None,
) -> bool:
    if not public_key_hex or not isinstance(trusted_registry, Mapping):
        return False
    try:
        actual = public_key_fingerprint(public_key_hex)
    except (TypeError, ValueError):
        return False
    expected = trusted_registry.get(principal)
    return isinstance(expected, str) and hmac.compare_digest(expected, actual)


def _iso_z(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_intent(
    registry: KeyRegistry,
    *,
    user_id: str,
    category: str,
    budget_minor: int,
    mcc: list[str],
    prompt_playback: str,
    signed_at: datetime | str,
    ttl_seconds: int = 86_400,
    payee_allowlist: list[str] | None = None,
) -> dict[str, Any]:
    signed_at_value = _iso_z(signed_at)
    constraint_obj = {
        "category": category,
        "max_amount_minor": budget_minor,
        "mcc": sorted(mcc),
        "payee_allowlist": sorted(payee_allowlist or []),
        "prompt_playback": prompt_playback,
        "signed_at": signed_at_value,
        "ttl_seconds": ttl_seconds,
    }
    signature, public_key = registry.sign(f"user:{user_id}", constraint_obj)
    return {
        "category": category,
        "budget_minor": budget_minor,
        "ttl_seconds": ttl_seconds,
        "prompt_playback": prompt_playback,
        "constraint_obj": constraint_obj,
        "signed_at": signed_at_value,
        "user_id": user_id,
        "user_signature": signature,
        "signer_public_key": public_key,
        "signature_valid": True,
    }


def build_cart(
    registry: KeyRegistry,
    *,
    merchant_id: str,
    items: list[dict[str, Any]],
    payee_did: str,
    mcc: str,
) -> dict[str, Any]:
    signed_items = [dict(item) for item in items]
    signature, public_key = registry.sign(f"merchant:{merchant_id}", signed_items)
    return {
        "items": signed_items,
        "total_minor": sum(item["price_minor"] * item.get("qty", 1) for item in signed_items),
        "payee_did": payee_did,
        "mcc": mcc,
        "merchant_id": merchant_id,
        "merchant_signature": signature,
        "signer_public_key": public_key,
        "signature_valid": True,
    }


def cart_hash(cart: dict[str, Any]) -> str:
    stable_cart = {key: value for key, value in cart.items() if key != "signature_valid"}
    return "sha256:" + hashlib.sha256(canonical_json(stable_cart)).hexdigest()


def _payment_payload(payment: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cart_hash",
        "amount_minor",
        "presence_flag",
        "agent_id",
        "agent_registered",
        "delegation_depth",
    )
    return {key: payment[key] for key in keys}


def build_payment(
    registry: KeyRegistry,
    *,
    cart: dict[str, Any],
    agent_id: str,
    presence_flag: str,
    delegation_depth: int,
    agent_registered: bool = True,
) -> dict[str, Any]:
    payment = {
        "cart_hash": cart_hash(cart),
        "hash_matches_cart": True,
        "amount_minor": cart["total_minor"],
        "presence_flag": presence_flag,
        "agent_id": agent_id,
        "agent_registered": agent_registered,
        "delegation_depth": delegation_depth,
    }
    signature, public_key = registry.sign(f"agent:{agent_id}", _payment_payload(payment))
    payment.update(
        agent_signature=signature,
        signer_public_key=public_key,
        signature_valid=True,
    )
    return payment


def resign_cart(registry: KeyRegistry, cart: dict[str, Any]) -> None:
    signature, public_key = registry.sign(f"merchant:{cart['merchant_id']}", cart["items"])
    cart.update(merchant_signature=signature, signer_public_key=public_key, signature_valid=True)


def resign_intent(registry: KeyRegistry, intent: dict[str, Any]) -> None:
    signature, public_key = registry.sign(f"user:{intent['user_id']}", intent["constraint_obj"])
    intent.update(user_signature=signature, signer_public_key=public_key, signature_valid=True)


def resign_payment(registry: KeyRegistry, payment: dict[str, Any]) -> None:
    signature, public_key = registry.sign(f"agent:{payment['agent_id']}", _payment_payload(payment))
    payment.update(agent_signature=signature, signer_public_key=public_key, signature_valid=True)


def relink_payment(registry: KeyRegistry, chain: dict[str, Any], *, match_amount: bool = True) -> None:
    payment = chain["payment"]
    payment["cart_hash"] = cart_hash(chain["cart"])
    payment["hash_matches_cart"] = True
    if match_amount:
        payment["amount_minor"] = chain["cart"]["total_minor"]
    resign_payment(registry, payment)


def build_chain(
    registry: KeyRegistry,
    *,
    user_id: str,
    merchant_id: str,
    agent_id: str,
    category: str,
    budget_minor: int,
    mcc: str,
    prompt_playback: str,
    signed_at: datetime | str,
    items: list[dict[str, Any]],
    payee_did: str,
    presence_flag: str,
    delegation_depth: int,
    ttl_seconds: int = 86_400,
) -> dict[str, Any]:
    intent = build_intent(
        registry,
        user_id=user_id,
        category=category,
        budget_minor=budget_minor,
        mcc=[mcc],
        prompt_playback=prompt_playback,
        signed_at=signed_at,
        ttl_seconds=ttl_seconds,
        payee_allowlist=[payee_did],
    )
    cart = build_cart(
        registry,
        merchant_id=merchant_id,
        items=items,
        payee_did=payee_did,
        mcc=mcc,
    )
    return {
        "intent": intent,
        "cart": cart,
        "payment": build_payment(
            registry,
            cart=cart,
            agent_id=agent_id,
            presence_flag=presence_flag,
            delegation_depth=delegation_depth,
        ),
    }


def verify_chain(
    chain: dict[str, Any],
    trusted_registry: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    intent = chain["intent"]
    cart = chain["cart"]
    payment = chain["payment"]
    constraints = intent["constraint_obj"]
    signed_constraints_complete = SIGNED_CONSTRAINT_FIELDS <= constraints.keys()
    return {
        "intent_signature_valid": verify_signature(
            constraints, intent.get("user_signature"), intent.get("signer_public_key")
        ),
        "cart_signature_valid": verify_signature(
            cart["items"], cart.get("merchant_signature"), cart.get("signer_public_key")
        ),
        "payment_signature_valid": verify_signature(
            _payment_payload(payment), payment.get("agent_signature"), payment.get("signer_public_key")
        ),
        "intent_registration_valid": _registration_valid(
            f"user:{intent.get('user_id', '')}",
            intent.get("signer_public_key"),
            trusted_registry,
        ),
        "cart_registration_valid": _registration_valid(
            f"merchant:{cart.get('merchant_id', '')}",
            cart.get("signer_public_key"),
            trusted_registry,
        ),
        "payment_registration_valid": _registration_valid(
            f"agent:{payment.get('agent_id', '')}",
            payment.get("signer_public_key"),
            trusted_registry,
        ),
        "signed_constraints_complete": signed_constraints_complete,
        "hash_matches_cart": payment["cart_hash"] == cart_hash(cart),
        "amount_matches_cart": payment["amount_minor"] == cart["total_minor"],
        "playback_matches_signed_constraint": signed_constraints_complete
        and intent.get("prompt_playback") == constraints["prompt_playback"],
        "intent_metadata_matches_signed_constraint": (
            signed_constraints_complete
            and intent.get("category") == constraints["category"]
            and intent.get("budget_minor") == constraints["max_amount_minor"]
            and intent.get("signed_at") == constraints["signed_at"]
            and intent.get("ttl_seconds") == constraints["ttl_seconds"]
        ),
    }


def annotate_chain(chain: dict[str, Any]) -> dict[str, bool]:
    checks = verify_chain(chain)
    chain["intent"]["signature_valid"] = checks["intent_signature_valid"]
    chain["cart"]["signature_valid"] = checks["cart_signature_valid"]
    chain["payment"]["signature_valid"] = checks["payment_signature_valid"]
    chain["payment"]["hash_matches_cart"] = checks["hash_matches_cart"]
    return checks
