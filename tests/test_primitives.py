import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from defend.embeddings import CachedMiniLM
from defend.features import FeatureExtractor
from sim.attacks import all_attacks, apply_attack
from sim.attacks.catalog import ATTACKS
from sim.mandate_twin import verify_chain
from sim.population import PopulationSimulator
from sim.primitives import LIFECYCLE, has_signal


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "contracts" / "event.schema.json").read_text(encoding="utf-8"))


def test_twelve_attacks_emit_their_intended_signals_and_contract() -> None:
    attacks = all_attacks(include_heldout=True)
    simulator = PopulationSimulator(seed=13, customers=80, merchants_per_category=1)
    validator = Draft202012Validator(SCHEMA, format_checker=FormatChecker())

    assert len(attacks) == 12
    assert len({attack.name for attack in attacks}) == 12
    for attack in attacks:
        base = simulator.generate_legitimate(force_channel=attack.base_channel)
        signed_intent = copy.deepcopy((base.get("mandate_chain") or {}).get("intent"))
        event = apply_attack(base, attack, registry=simulator.registry)
        assert not list(validator.iter_errors(event))
        assert event["label"]["attack_primitives"] == list(attack.primitives)
        assert event["label"]["attack_family"] == attack.family
        assert event["label"]["split"] == attack.split
        assert all(has_signal(event, signal) for signal in attack.intended_signals)
        if attack.family == "budget_ceiling_capture":
            assert event["mandate_chain"]["intent"] == signed_intent
            assert event["txn"]["amount_minor"] <= signed_intent["constraint_obj"]["max_amount_minor"]
        if event["mandate_chain"]:
            checks = verify_chain(
                event["mandate_chain"], simulator.registry.public_key_fingerprints()
            )
            assert checks["intent_signature_valid"]
            assert checks["payment_signature_valid"]
            if attack.family == "cart_hash_decoupling":
                assert not checks["hash_matches_cart"]


def test_novel_compositions_and_heldout_primitives_are_separated() -> None:
    all_definitions = all_attacks(include_heldout=True)
    training = [attack for attack in all_definitions if attack.split in {"A", "B"}]
    novel = [attack for attack in all_definitions if attack.split == "C"]
    heldout = [attack for attack in all_definitions if attack.split == "D"]
    training_paths = {attack.primitives for attack in training}
    training_primitives = {primitive for attack in training for primitive in attack.primitives}

    assert all(attack.primitives not in training_paths for attack in novel)
    assert all(set(attack.primitives) <= training_primitives for attack in novel)
    assert {"delayed_exec", "cash_out"}.isdisjoint(training_primitives)
    assert {"delayed_exec", "cash_out"} <= {primitive for attack in heldout for primitive in attack.primitives}
    assert all(
        [LIFECYCLE.index(primitive) for primitive in attack.primitives]
        == sorted(LIFECYCLE.index(primitive) for primitive in attack.primitives)
        for attack in all_definitions
    )


def test_heldout_generators_are_physically_isolated() -> None:
    heldout_dir = ROOT / "sim" / "attacks" / "heldout"
    modules = {path.name for path in heldout_dir.glob("*.py")}

    assert {"ttl_dormancy.py", "bust_out.py"} <= modules
    assert all(attack.split != "D" for attack in ATTACKS)


def test_contract_rejects_invalid_time_fractional_money_and_incomplete_intent() -> None:
    event = next(
        value
        for value in map(json.loads, (ROOT / "sim" / "fixtures" / "events.jsonl").read_text(encoding="utf-8").splitlines())
        if value["mandate_chain"]
    )
    validator = Draft202012Validator(SCHEMA, format_checker=FormatChecker())

    invalid_time = copy.deepcopy(event)
    invalid_time["ts"] = "not-a-date"
    fractional_money = copy.deepcopy(event)
    fractional_money["txn"]["amount_minor"] = 1.25
    incomplete_intent = copy.deepcopy(event)
    incomplete_intent["mandate_chain"]["intent"].pop("prompt_playback")

    assert list(validator.iter_errors(invalid_time))
    assert list(validator.iter_errors(fractional_money))
    assert list(validator.iter_errors(incomplete_intent))


def test_contract_requires_mandate_chain_only_for_agentic_checkout() -> None:
    event = next(
        value
        for value in map(
            json.loads,
            (ROOT / "sim" / "fixtures" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if value["mandate_chain"]
    )
    validator = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
    missing_agentic_chain = copy.deepcopy(event)
    missing_agentic_chain["mandate_chain"] = None
    non_agentic = copy.deepcopy(missing_agentic_chain)
    non_agentic.update(channel="card_ecom", rail="card", agent_trace=None)

    assert list(validator.iter_errors(missing_agentic_chain))
    assert not list(validator.iter_errors(non_agentic))


def test_each_attack_changes_a_real_blue_feature(tmp_path: Path) -> None:
    mappings = {
        "hidden_content_provenance": ("injection_fingerprint_score", 1),
        "intent_cart_semantic_drift": ("intent_cart_semantic_drift", 1),
        "price_vs_market_median": ("price_vs_market_median", 1),
        "playback_constraint_mismatch": ("playback_constraint_consistency", -1),
        "ttl_expired": ("ttl_utilisation_pct", 1),
        "presence_evidence_mismatch": ("present_flag_vs_behavioral_evidence", -1),
        "intent_fanout": ("fanout_count_per_intent", 1),
        "cart_hash_mismatch": ("hash_matches_cart", -1),
        "delegation_scope_violation": ("delegation_depth", 1),
        "merchant_registry_miss": ("signer_did_in_registry", -1),
        "card_testing_velocity": ("velocity_1h", 1),
        "bust_out_regime_change": ("credit_utilisation_regime_change", 1),
        "collect_direction_mismatch": ("payment_direction_consistency", -1),
    }
    simulator = PopulationSimulator(seed=37, customers=80, merchants_per_category=1)
    pairs: list[tuple[object, dict, dict]] = []
    events: list[dict] = []
    for attack in all_attacks(include_heldout=True):
        base = simulator.generate_legitimate(force_channel=attack.base_channel)
        mutated = apply_attack(base, attack, registry=simulator.registry)
        pairs.append((attack, base, mutated))
        events.extend((base, mutated))

    frame = FeatureExtractor(
        CachedMiniLM(tmp_path / "embeddings.npz"),
        trusted_registry=simulator.registry.public_key_fingerprints(),
    ).transform(events).frame
    for index, (attack, _base, _mutated) in enumerate(pairs):
        changed = False
        for signal in attack.intended_signals:
            feature, direction = mappings[signal]
            delta = frame.loc[index * 2 + 1, feature] - frame.loc[index * 2, feature]
            changed |= delta * direction > 1e-9
        assert changed, f"{attack.family} did not change an intended Blue feature"
