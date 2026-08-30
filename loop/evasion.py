"""Seeded non-operational mutation search against training-fold events only."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from defend.harness import assert_training_partitions


@dataclass(frozen=True)
class Candidate:
    event: dict[str, Any]
    mutation: str
    parent_id: str


def fitness(payoff: float, blue_score: float) -> float:
    return max(0.0, float(payoff)) * (1 - min(1.0, max(0.0, float(blue_score))))


def _soften_page(event: dict[str, Any]) -> None:
    pages = (event.get("agent_trace") or {}).get("pages_read") or []
    for page in pages:
        page["hidden_text_present"] = False
        page["invisible_css"] = False
        page["zero_width_chars"] = max(0, int(page.get("zero_width_chars") or 0) // 4)
        page["hidden_text_sample"] = "[TRUNCATED / NON-OPERATIONAL EVASION SAMPLE]"


def _blend_topics(event: dict[str, Any]) -> None:
    chain = event.get("mandate_chain") or {}
    category = str((chain.get("intent") or {}).get("category") or "shopping").replace("_", " ")
    for call in (event.get("agent_trace") or {}).get("tool_calls") or []:
        call["topic"] = category


def _slow_negotiation(event: dict[str, Any]) -> None:
    trace = event.get("agent_trace") or {}
    trace["negotiation_seconds"] = max(12.0, float(trace.get("negotiation_seconds") or 0) * 2.5)


def _reduce_fanout(event: dict[str, Any]) -> None:
    txn = event.get("txn") or {}
    txn["fanout_count_per_intent"] = 1


def _reduce_graph_signal(event: dict[str, Any]) -> None:
    txn = event.get("txn") or {}
    txn["shared_device_cluster_size"] = 1
    txn["connected_component_size"] = 1


def _pace_velocity(event: dict[str, Any]) -> None:
    txn = event.get("txn") or {}
    txn["authorisation_attempts_10m"] = 1
    txn["authorisation_attempts_24h"] = 1
    txn["cards_tested_10m"] = 1
    txn["cashout_velocity_1h"] = 1


def _reuse_payee(event: dict[str, Any]) -> None:
    txn = event.get("txn") or {}
    txn["is_new_beneficiary"] = False
    txn["beneficiary_in_degree"] = min(2, int(txn.get("beneficiary_in_degree") or 0))


MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("artifact-softening", _soften_page),
    ("topic-blending", _blend_topics),
    ("slow-negotiation", _slow_negotiation),
    ("fanout-reduction", _reduce_fanout),
    ("graph-signal-reduction", _reduce_graph_signal),
    ("paced-velocity", _pace_velocity),
    ("known-payee-reuse", _reuse_payee),
)


def mutate_training_events(
    source_events: Iterable[dict[str, Any]],
    *,
    round_number: int,
    count: int = 24,
    seed: int = 20260827,
) -> list[Candidate]:
    sources = list(source_events)
    assert_training_partitions(sources)
    fraudulent = [event for event in sources if bool(event["label"]["is_fraud"])]
    if not fraudulent:
        raise ValueError("evasion search needs fraudulent A/B training examples")
    rng = random.Random(seed + round_number)
    output = []
    for number in range(count):
        parent = rng.choice(fraudulent)
        event = copy.deepcopy(parent)
        chosen = rng.sample(MUTATIONS, k=rng.randint(1, min(3, len(MUTATIONS))))
        for _, mutation in chosen:
            mutation(event)
        event["event_id"] = f"evt_arena_r{round_number}_{number:04d}"
        event["label"]["split"] = "B"
        event["label"]["arena_round"] = round_number
        event["scores"] = {name: None for name in ("semantic", "provenance", "temporal", "economic", "structural", "fused")}
        event["decision"] = {"action": None, "legacy_rules_action": None, "reason_codes": [], "latency_ms": None}
        output.append(Candidate(event, "+".join(name for name, _ in chosen), str(parent["event_id"])))
    return output
