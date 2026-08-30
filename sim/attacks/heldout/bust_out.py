"""Held-out trust-grooming and cash-out family."""

from sim.primitives import AttackDefinition


ATTACK = AttackDefinition(
    atlas_id="GPF-C04",
    name="synthetic identity bust-out",
    family="synthetic_identity_bust_out",
    split="D",
    primitives=("reconnaissance", "payment_manip", "cash_out"),
    intended_signals=("bust_out_regime_change",),
    base_channel="card_ecom",
    parameters={
        "account_age_days": 420,
        "prior_utilisation_pct": 0.08,
        "current_utilisation_pct": 0.98,
        "cashout_velocity": 11,
    },
)
