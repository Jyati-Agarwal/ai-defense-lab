from defend.policy import action_for_score, legacy_rules_action, policy_decision


def test_policy_thresholds_are_exact() -> None:
    assert action_for_score(0.299999) == "ALLOW"
    assert action_for_score(0.30) == "STEP_UP"
    assert action_for_score(0.60) == "HOLD"
    assert action_for_score(0.85) == "BLOCK"


def test_legacy_baseline_and_reasons() -> None:
    event = {
        "txn": {"amount_minor": 7_800_000, "blocklisted_bin": False, "geo_velocity_kmh": 0},
        "mandate_chain": None,
    }
    assert legacy_rules_action(event) == "STEP_UP"
    decision = policy_decision(event, 0.9, ["Measured TreeSHAP reason"], 1.25)
    assert decision["action"] == "BLOCK"
    assert decision["reason_codes"] == ["Measured TreeSHAP reason"]
