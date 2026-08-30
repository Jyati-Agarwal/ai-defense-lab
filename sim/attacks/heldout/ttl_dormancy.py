"""Held-out long-horizon execution family."""

from sim.primitives import AttackDefinition


ATTACK = AttackDefinition(
    atlas_id="GPF-A20",
    name="TTL dormancy execution",
    family="ttl_dormancy",
    split="D",
    primitives=("intent_manip", "delayed_exec"),
    intended_signals=("ttl_expired",),
    parameters={"delay_seconds": 90_000, "shift_topic": "deferred execution"},
)
