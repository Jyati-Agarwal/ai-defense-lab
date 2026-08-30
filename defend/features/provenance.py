"""Observable retrieval and tool-use provenance features."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import numpy as np


INSTRUCTION_PATTERN = re.compile(
    r"\b(ignore|override|previous instructions|system message|transfer|payment mandate)\b",
    re.IGNORECASE,
)


def topics(event: dict[str, Any]) -> list[str]:
    trace = event.get("agent_trace") or {}
    return [str(call.get("topic") or "") for call in trace.get("tool_calls") or [] if call.get("topic")]


def domains(event: dict[str, Any]) -> list[str]:
    trace = event.get("agent_trace") or {}
    return [
        urlparse(str(page.get("url") or "")).netloc.lower()
        for page in trace.get("pages_read") or []
        if page.get("url")
    ]


def provenance_features(
    event: dict[str, Any],
    topic_vectors: np.ndarray,
    known_domains: set[str],
    negotiation_mean: float,
    negotiation_std: float,
) -> dict[str, float]:
    trace = event.get("agent_trace") or {}
    pages = trace.get("pages_read") or []
    hidden = any(bool(page.get("hidden_text_present")) for page in pages)
    invisible = any(bool(page.get("invisible_css")) for page in pages)
    zero_width = sum(max(0, int(page.get("zero_width_chars") or 0)) for page in pages)
    samples = " ".join(str(page.get("hidden_text_sample") or "") for page in pages)
    instruction = bool(INSTRUCTION_PATTERN.search(samples))
    fingerprint = min(1.0, 0.35 * hidden + 0.25 * invisible + min(0.25, zero_width / 40) + 0.25 * instruction)
    if len(topic_vectors) < 2:
        topic_shift = 0.0
    else:
        topic_shift = float(np.clip(np.max(1 - np.sum(topic_vectors[:-1] * topic_vectors[1:], axis=1)), 0, 1))
    observed_domains = domains(event)
    novelty = 0.0 if not observed_domains else sum(domain not in known_domains for domain in observed_domains) / len(observed_domains)
    seconds = float(trace.get("negotiation_seconds") or negotiation_mean)
    speed_z = max(0.0, (negotiation_mean - seconds) / max(negotiation_std, 1.0))
    return {
        "injection_fingerprint_score": fingerprint,
        "tool_call_topic_shift": topic_shift,
        "retrieved_domain_novelty": float(novelty),
        "negotiation_speed_zscore": min(speed_z, 10.0),
    }
