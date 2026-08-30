from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


class ArtifactMissing(FileNotFoundError):
    pass


class ArtifactInvalid(ValueError):
    pass


class ArtifactStore:
    """Read immutable, precomputed demo artifacts once per process."""

    def __init__(self, project_root: Path, artifact_sha256: Mapping[str, str]):
        self.project_root = project_root
        self.results = project_root / "results"
        self.artifact_sha256 = dict(artifact_sha256)

    @lru_cache(maxsize=32)
    def _verified_bytes(self, filename: str) -> bytes:
        if Path(filename).name != filename:
            raise ArtifactInvalid("artifact filename must not contain a path")
        path = self.results / filename
        expected = self.artifact_sha256.get(filename)
        if expected is None:
            raise ArtifactInvalid(f"{filename} is not bound by the verified release")
        if not path.is_file():
            raise ArtifactMissing(path)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ArtifactMissing(path) from error
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ArtifactInvalid(f"{filename} no longer matches the verified release")
        return payload

    @lru_cache(maxsize=32)
    def json(self, name: str) -> Any:
        filename = f"{name}.json"
        try:
            return json.loads(self._verified_bytes(filename))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactInvalid(f"{filename} is not valid JSON") from error

    @lru_cache(maxsize=4)
    def scored_events(self) -> tuple[dict[str, Any], ...]:
        filename = "scored_events.jsonl"
        path = self.results / filename

        events: list[dict[str, Any]] = []
        try:
            lines = self._verified_bytes(filename).decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ArtifactInvalid(f"{filename} is not UTF-8") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                scores = event["scores"]
                decision = event["decision"]
                valid = (
                    isinstance(event, dict)
                    and isinstance(event.get("event_id"), str)
                    and bool(event["event_id"])
                    and isinstance(scores, dict)
                    and isinstance(scores.get("fused"), (int, float))
                    and isinstance(decision, dict)
                    and isinstance(decision.get("action"), str)
                    and bool(decision["action"])
                    and isinstance(decision.get("reason_codes"), list)
                    and bool(decision["reason_codes"])
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                valid = False
            if not valid:
                raise ArtifactInvalid(f"{path}:{line_number} is not a fully scored event")
            events.append(event)
        if not events:
            raise ArtifactInvalid(f"{path} contains no scored events")
        return tuple(events)

    def event(self, event_id: str) -> dict[str, Any] | None:
        events = self.scored_events()
        if event_id == "evt_flagship":
            source = next(
                (
                    event
                    for event in events
                    if event.get("event_id") != event_id
                    and (event.get("label") or {}).get("attack_family") == "injection_to_payment"
                    and event["decision"].get("legacy_rules_action") == "ALLOW"
                    and event["decision"].get("action") == "BLOCK"
                ),
                None,
            )
            if source is not None:
                return {**source, "source_event_id": source["event_id"], "event_id": event_id}
            return None
        return next((event for event in events if event.get("event_id") == event_id), None)

    def precedent(self, event_id: str) -> dict[str, Any] | None:
        precedents = self.json("precedents")
        if not isinstance(precedents, (dict, list)):
            return None
        values = precedents.values() if isinstance(precedents, dict) else precedents
        if event_id != "evt_borderline":
            if isinstance(precedents, dict):
                result = precedents.get(event_id)
                return result if isinstance(result, dict) else None
            return next(
                (
                    precedent
                    for precedent in precedents
                    if isinstance(precedent, dict) and precedent.get("event_id") == event_id
                ),
                None,
            )

        events = {event.get("event_id"): event for event in self.scored_events()}
        candidates = [
            precedent
            for precedent in values
            if isinstance(precedent, dict)
            and precedent.get("event_id") != event_id
            and precedent.get("source_event_id") is None
            and precedent.get("verdict") == "suppressed"
            and (events.get(precedent.get("event_id"), {}).get("label") or {}).get("is_fraud") is False
            and isinstance(precedent.get("precedent_support"), (int, float))
        ]
        if not candidates:
            return None
        source = max(candidates, key=lambda precedent: precedent["precedent_support"])
        return {**source, "source_event_id": source["event_id"], "event_id": event_id}
